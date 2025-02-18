import torch
import torch.nn as nn
from inspect import isfunction
import copy
import einops
from einops import rearrange, repeat
from typing import Callable, Optional, List
from human_models.human_models import SMPLX, SMPL
from functools import partial
from timm.layers import drop_path, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class TransformerDecoderHead(nn.Module):
    """ Cross-attention based Transformer decoder
    """

    def __init__(self,
            feat_dim=1080,
            dim_out=512,
            task_tokens_num=80):
        super().__init__()

        # TODO: put args in config file
        self.dim = feat_dim
        self.dim_out = dim_out
        self.token_dim = task_tokens_num

        smpl_x = SMPLX()
        HAND_JOINT_NUM = len(smpl_x.orig_joint_part['rhand'])        
        BODY_JOINT_NUM = len(smpl_x.orig_joint_part['body'])
        SHAPE_NUM = smpl_x.shape_param_dim
        EXPRESSION_NUM = smpl_x.expr_code_dim


        transformer_args = dict(
            num_tokens=1,
            token_dim=self.token_dim,
            dim=self.dim,
            depth=6,
            heads=8,
            mlp_dim=self.dim,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
            norm='layer',
            context_dim=self.dim
        )
        self.transformer = TransformerDecoder(
            **transformer_args
        )

        # [b, token_dim, dim] -> [b, token_dim, dim_out]
        self.token_conv = nn.Linear(self.dim, self.dim_out)
        # heads
        # body
        self.dec_body_root_pose = nn.Linear(1*self.dim_out, 6) # 1 [b, 6]
        self.dec_body_pose = nn.Linear((BODY_JOINT_NUM-1)*self.dim_out, (BODY_JOINT_NUM-1)*6) # 21 [b, 21 ,6]
        self.dec_body_shape = nn.Linear(SHAPE_NUM*self.dim_out, SHAPE_NUM) # 10 [b, 10]
        self.dec_body_cam = nn.Linear(1*self.dim_out, 3) # 1 [b, 3]

        # left and right hand
        self.dec_hand_root_pose = nn.Linear(2*self.dim_out, 2*6) # 2 [b, 2, 6]
        self.dec_hand_pose = nn.Linear(2*HAND_JOINT_NUM*self.dim_out, 2*HAND_JOINT_NUM*6) # 30 [b, 30, 6]
        self.dec_hand_cam = nn.Linear(2*self.dim_out, 2*3) # 2 [b, 2, 3]

        # face
        self.dec_face_root_pose = nn.Linear(1*self.dim_out, 6) # 1 [b, 6]
        self.dec_face_expression = nn.Linear(EXPRESSION_NUM*self.dim_out, EXPRESSION_NUM) # 10 [b, 10]
        self.dec_face_jaw_pose = nn.Linear(1*self.dim_out, 6)  # 1 [b, 6]
        self.dec_face_cam = nn.Linear(1*self.dim_out, 3)   # 1 [b, 3]


    def forward(self, token, x, **kwargs):
        
        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        token = torch.cat((token, x), dim=1)  # Concatenated input to decoder

        # Pass through transformer
        token_out = self.transformer(token, context=x)
        token_out = self.token_conv(token_out)
        token_out = token_out[:, :self.token_dim, :] # (B, C)

        # Readout from token_out
        token_body_root = token_out[:, :1, :].view(batch_size, -1)
        token_body_pose = token_out[:, 1:22, :].view(batch_size, -1)
        token_body_shape = token_out[:, 22:32, :].view(batch_size, -1)
        token_body_cam = token_out[:, 32:33, :].view(batch_size, -1)

        token_hand_root = token_out[:, 33:35, :].view(batch_size, -1)
        token_hand_pose = token_out[:, 35:65, :].view(batch_size, -1)
        token_hand_cam = token_out[:, 65:67, :].view(batch_size, -1)

        token_face_root = token_out[:, 67:68, :].view(batch_size, -1)
        token_face_expression = token_out[:, 68:78, :].view(batch_size, -1)
        token_face_jaw = token_out[:, 78:79, :].view(batch_size, -1)
        token_face_cam = token_out[:, 79:80, :].view(batch_size, -1)

        # Decode
        pred_body_root_pose = self.dec_body_root_pose(token_body_root) # 1
        pred_body_pose = self.dec_body_pose(token_body_pose) # 21
        pred_body_betas = self.dec_body_shape(token_body_shape) # 10
        pred_body_cam = self.dec_body_cam(token_body_cam) # 1

        pred_hand_root_pose = self.dec_hand_root_pose(token_hand_root) # 2
        pred_hand_pose = self.dec_hand_pose(token_hand_pose) # 30
        pred_hand_cam = self.dec_hand_cam(token_hand_cam) # 2

        pred_face_root_pose = self.dec_face_root_pose(token_face_root) # 1
        pred_face_expression = self.dec_face_expression(token_face_expression) # 10
        pred_face_jaw_pose = self.dec_face_jaw_pose(token_face_jaw) # 1
        pred_face_cam = self.dec_face_cam(token_face_cam) # 1

        # all rotations in rot6d
        pred_params = {'body_root_pose': pred_body_root_pose,
                        'body_pose': pred_body_pose,
                        'body_betas': pred_body_betas,
                        'body_cam': pred_body_cam,
                        'lhand_root_pose': pred_hand_root_pose[:, :6],
                        'rhand_root_pose': pred_hand_root_pose[:, 6:],
                        'lhand_pose': pred_hand_pose[:, :90],
                        'rhand_pose': pred_hand_pose[:, 90:],
                        'lhand_cam': pred_hand_cam[:, :3],
                        'rhand_cam': pred_hand_cam[:, 3:],
                        'face_root_pose': pred_face_root_pose,
                        'face_expression': pred_face_expression,
                        'face_jaw_pose': pred_face_jaw_pose,
                        'face_cam': pred_face_cam}

        return pred_params

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = 'drop',
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
        skip_token_embedding: bool = False,
    ):
        super().__init__()
        if not skip_token_embedding:
            self.to_token_embedding = nn.Linear(token_dim, dim)
        else:
            self.to_token_embedding = nn.Identity()
            if token_dim != dim:
                raise ValueError(
                    f"token_dim ({token_dim}) != dim ({dim}) when skip_token_embedding is True"
                )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)
        elif emb_dropout_type == "normal":
            self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerCrossAttn(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            norm=norm,
            norm_cond_dim=norm_cond_dim,
            context_dim=context_dim,
        )

    def forward(self, x: torch.Tensor, *args, context=None, context_list=None):
        b, n, _ = x.shape

        x = self.dropout(x)
        x += self.pos_embedding[:, :n]

        x = self.transformer(x, *args, context=context, context_list=context_list)
        return x

class TransformerCrossAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ca = CrossAttention(
                dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout
            )
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ca, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, *args, context=None, context_list=None):
        if context_list is None:
            context_list = [context] * len(self.layers)
        if len(context_list) != len(self.layers):
            raise ValueError(f"len(context_list) != len(self.layers) ({len(context_list)} != {len(self.layers)})")

        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            x = self_attn(x, *args) + x
            x = cross_attn(x, *args, context=context_list[i]) + x
            x = ff(x, *args) + x
        return x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable, norm: str = "layer", norm_cond_dim: int = -1):
        super().__init__()
        self.norm = normalization_layer(norm, dim, norm_cond_dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if isinstance(self.norm, AdaptiveLayerNorm1D):
            return self.fn(self.norm(x, *args), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class DropTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, dim)
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[0, :, 0], self.p).bernoulli().bool()
            # TODO: permutation idx for each batch using torch.argsort
            if zero_mask.any():
                x = x[:, ~zero_mask, :]
        return x


class ZeroTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, dim)
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[:, :, 0], self.p).bernoulli().bool()
            # Zero-out the masked tokens
            x[zero_mask, :] = 0
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        context_dim = default(context_dim, dim)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, context=None):
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        return out
    

class AdaptiveLayerNorm1D(torch.nn.Module):
    def __init__(self, data_dim: int, norm_cond_dim: int):
        super().__init__()
        if data_dim <= 0:
            raise ValueError(f"data_dim must be positive, but got {data_dim}")
        if norm_cond_dim <= 0:
            raise ValueError(f"norm_cond_dim must be positive, but got {norm_cond_dim}")
        self.norm = torch.nn.LayerNorm(
            data_dim
        )  # TODO: Check if elementwise_affine=True is correct
        self.linear = torch.nn.Linear(norm_cond_dim, 2 * data_dim)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (batch, ..., data_dim)
        # t: (batch, norm_cond_dim)
        # return: (batch, data_dim)
        x = self.norm(x)
        alpha, beta = self.linear(t).chunk(2, dim=-1)

        # Add singleton dimensions to alpha and beta
        if x.dim() > 2:
            alpha = alpha.view(alpha.shape[0], *([1] * (x.dim() - 2)), alpha.shape[1])
            beta = beta.view(beta.shape[0], *([1] * (x.dim() - 2)), beta.shape[1])

        return x * (1 + alpha) + beta


class SequentialCond(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self:
            if isinstance(module, (AdaptiveLayerNorm1D, SequentialCond, ResidualMLPBlock)):
                # print(f'Passing on args to {module}', [a.shape for a in args])
                input = module(input, *args, **kwargs)
            else:
                # print(f'Skipping passing args to {module}', [a.shape for a in args])
                input = module(input)
        return input


def normalization_layer(norm: Optional[str], dim: int, norm_cond_dim: int = -1):
    if norm == "batch":
        return torch.nn.BatchNorm1d(dim)
    elif norm == "layer":
        return torch.nn.LayerNorm(dim)
    elif norm == "ada":
        assert norm_cond_dim > 0, f"norm_cond_dim must be positive, got {norm_cond_dim}"
        return AdaptiveLayerNorm1D(dim, norm_cond_dim)
    elif norm is None:
        return torch.nn.Identity()
    else:
        raise ValueError(f"Unknown norm: {norm}")


def linear_norm_activ_dropout(
    input_dim: int,
    output_dim: int,
    activation: torch.nn.Module = torch.nn.ReLU(),
    bias: bool = True,
    norm: Optional[str] = "layer",  # Options: ada/batch/layer
    dropout: float = 0.0,
    norm_cond_dim: int = -1,
) -> SequentialCond:
    layers = []
    layers.append(torch.nn.Linear(input_dim, output_dim, bias=bias))
    if norm is not None:
        layers.append(normalization_layer(norm, output_dim, norm_cond_dim))
    layers.append(copy.deepcopy(activation))
    if dropout > 0.0:
        layers.append(torch.nn.Dropout(dropout))
    return SequentialCond(*layers)


def create_simple_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: torch.nn.Module = torch.nn.ReLU(),
    bias: bool = True,
    norm: Optional[str] = "layer",  # Options: ada/batch/layer
    dropout: float = 0.0,
    norm_cond_dim: int = -1,
) -> SequentialCond:
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            linear_norm_activ_dropout(
                prev_dim, hidden_dim, activation, bias, norm, dropout, norm_cond_dim
            )
        )
        prev_dim = hidden_dim
    layers.append(torch.nn.Linear(prev_dim, output_dim, bias=bias))
    return SequentialCond(*layers)


class ResidualMLPBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = True,
        norm: Optional[str] = "layer",  # Options: ada/batch/layer
        dropout: float = 0.0,
        norm_cond_dim: int = -1,
    ):
        super().__init__()
        if not (input_dim == output_dim == hidden_dim):
            raise NotImplementedError(
                f"input_dim {input_dim} != output_dim {output_dim} is not implemented"
            )

        layers = []
        prev_dim = input_dim
        for i in range(num_hidden_layers):
            layers.append(
                linear_norm_activ_dropout(
                    prev_dim, hidden_dim, activation, bias, norm, dropout, norm_cond_dim
                )
            )
            prev_dim = hidden_dim
        self.model = SequentialCond(*layers)
        self.skip = torch.nn.Identity()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.model(x, *args, **kwargs)


class ResidualMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = True,
        norm: Optional[str] = "layer",  # Options: ada/batch/layer
        dropout: float = 0.0,
        num_blocks: int = 1,
        norm_cond_dim: int = -1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.model = SequentialCond(
            linear_norm_activ_dropout(
                input_dim, hidden_dim, activation, bias, norm, dropout, norm_cond_dim
            ),
            *[
                ResidualMLPBlock(
                    hidden_dim,
                    hidden_dim,
                    num_hidden_layers,
                    hidden_dim,
                    activation,
                    bias,
                    norm,
                    dropout,
                    norm_cond_dim,
                )
                for _ in range(num_blocks)
            ],
            torch.nn.Linear(hidden_dim, output_dim, bias=bias),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.model(x, *args, **kwargs)


class FrequencyEmbedder(torch.nn.Module):
    def __init__(self, num_frequencies, max_freq_log2):
        super().__init__()
        frequencies = 2 ** torch.linspace(0, max_freq_log2, steps=num_frequencies)
        self.register_buffer("frequencies", frequencies)

    def forward(self, x):
        # x should be of size (N,) or (N, D)
        N = x.size(0)
        if x.dim() == 1:  # (N,)
            x = x.unsqueeze(1)  # (N, D) where D=1
        x_unsqueezed = x.unsqueeze(-1)  # (N, D, 1)
        scaled = self.frequencies.view(1, 1, -1) * x_unsqueezed  # (N, D, num_frequencies)
        s = torch.sin(scaled)
        c = torch.cos(scaled)
        embedded = torch.cat([s, c, x_unsqueezed], dim=-1).view(
            N, -1
        )  # (N, D * 2 * num_frequencies + D)
        return embedded

class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio),
                              padding=4 + 2 * (ratio // 2 - 1))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)
    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_head_dim=None
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention_ViT(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)
    
class Attention_ViT(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class ViT(torch.nn.Module):
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=80, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 hybrid_backbone=None, 
                 norm_layer=None, 
                 use_checkpoint=False,
                 frozen_stages=-1, 
                 ratio=1, 
                 last_norm=True,
                 patch_padding='pad', 
                 freeze_attn=False, 
                 freeze_ffn=False, 
                 task_tokens_num=80
                 ):
        # Protect mutable default arguments
        super(ViT, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth
        self.task_tokens_num = task_tokens_num

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches

        # task tokens for HPS estimation
        self.task_tokens = nn.Parameter(torch.zeros(1, task_tokens_num, embed_dim))
        trunc_normal_(self.task_tokens, std=.02)

        # since the pretraining model has class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            )
            for i in range(depth)])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False


    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        task_tokens = repeat(self.task_tokens, '() n d -> b n d', b=B)
        if self.pos_embed is not None:
            # fit for multiple GPU training
            # since the first element for pos embed (sin-cos manner) is zero, it will cause no difference
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        x = torch.cat((task_tokens, x), dim=1)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.last_norm(x)

        task_tokens = x[:, :self.task_tokens_num]  # [N,J,C]
        # task_tokens = torch.cat(task_tokens_, dim=-1)
        xp = x[:, self.task_tokens_num:]  # [N,Hp*Wp,C]

        xp = xp.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        return xp, task_tokens

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()
