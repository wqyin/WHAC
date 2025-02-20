import torch
import numpy as np
from scipy.ndimage import uniform_filter1d

from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer.cameras import look_at_rotation
from pytorch3d.utils import ico_sphere


def overlay_image_onto_background(image, mask, bbox, background):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    out_image = background.copy()
    bbox = bbox[0].int().cpu().numpy().copy()
    roi_image = out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    roi_image[mask] = image[mask]
    out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = roi_image

    return out_image


def update_intrinsics_from_bbox(K_org, bbox):
    device, dtype = K_org.device, K_org.dtype
    
    K = torch.zeros((K_org.shape[0], 4, 4)
    ).to(device=device, dtype=dtype)
    K[:, :3, :3] = K_org.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1
    
    image_sizes = []
    for idx, bbox in enumerate(bbox):
        left, upper, right, lower = bbox
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]

        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)
        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        K[idx, 0, 2] = new_cx
        K[idx, 1, 2] = new_cy
        image_sizes.append((int(new_height), int(new_width)))

    return K, image_sizes


def perspective_projection(x3d, K, R=None, T=None):
    if R != None:
        x3d = torch.matmul(R, x3d.transpose(1, 2)).transpose(1, 2)
    if T != None:
        x3d = x3d + T.transpose(1, 2)

    x2d = torch.div(x3d, x3d[..., 2:])
    x2d = torch.matmul(K, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    return x2d


def compute_bbox_from_points(X, img_w, img_h, scaleFactor=1.2):
    left = torch.clamp(X.min(1)[0][:, 0], min=0, max=img_w)
    right = torch.clamp(X.max(1)[0][:, 0], min=0, max=img_w)
    top = torch.clamp(X.min(1)[0][:, 1], min=0, max=img_h)
    bottom = torch.clamp(X.max(1)[0][:, 1], min=0, max=img_h)

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = (right - left)
    height = (bottom - top)

    new_left = torch.clamp(cx - width/2 * scaleFactor, min=0, max=img_w-1)
    new_right = torch.clamp(cx + width/2 * scaleFactor, min=1, max=img_w)
    new_top = torch.clamp(cy - height / 2 * scaleFactor, min=0, max=img_h-1)
    new_bottom = torch.clamp(cy + height / 2 * scaleFactor, min=1, max=img_h)

    bbox = torch.stack((new_left.detach(), new_top.detach(),
                        new_right.detach(), new_bottom.detach())).int().float().T
    
    return bbox


class Renderer():
    def __init__(self, width, height, focal_length, device, faces=None):

        self.width = width
        self.height = height
        self.focal_length = focal_length

        self.device = device
        if faces is not None:
            self.faces = torch.from_numpy(
                (faces).astype('int')
            ).unsqueeze(0).to(self.device)

        self.initialize_camera_params()
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -10.0]])
        self.create_renderer()
        self.human_trajectory_verts = []
        self.human_trajectory_faces = []
        self.human_trajectory_colors = []
    def create_renderer(self):
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=self.image_sizes[0],
                    blur_radius=1e-5,
                    max_faces_per_bin=30000),
            ),
            shader=SoftPhongShader(
                device=self.device,
                lights=self.lights,
            )
        )

    def create_camera(self, R=None, T=None):
        if R is not None:
            self.R = R.clone().view(1, 3, 3).to(self.device)
        if T is not None:
            self.T = T.clone().view(1, 3).to(self.device)

        return PerspectiveCameras(
            device=self.device,
            R=self.R.mT,
            T=self.T,
            K=self.K_full,
            image_size=self.image_sizes,
            in_ndc=False)


    def initialize_camera_params(self):
        """Hard coding for camera parameters
        TODO: Do some soft coding"""

        # Extrinsics
        self.R = torch.diag(
            torch.tensor([1, 1, 1])
        ).float().to(self.device).unsqueeze(0)

        self.T = torch.tensor(
            [0, 0, 0]
        ).unsqueeze(0).float().to(self.device)

        # Intrinsics
        self.K = torch.tensor(
            [[self.focal_length, 0, self.width/2],
            [0, self.focal_length, self.height/2],
            [0, 0, 1]]
        ).unsqueeze(0).float().to(self.device)
        self.bboxes = torch.tensor([[0, 0, self.width, self.height]]).float()
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
        self.cameras = self.create_camera()

    def set_cam_mesh(self, extrinsics, radius=0.1, height=0.2, up="y"):
        # set sphere geometry
        verts, faces, face_colors = camera_marker_geometry(radius, height, up)

        verts = verts.repeat(extrinsics.shape[0], 1, 1).to('cuda')
        verts = torch.einsum('bij,bkj->bki', extrinsics[:,:3,:3], verts) + extrinsics[:,:3,3][:,None,:]
        self.cam_geometry = [[verts[i], faces, face_colors] for i in range(extrinsics.shape[0])]

    def set_cam_trajectory(self, id, radius=0.015):
        points = torch.stack([self.cam_geometry[i][0][4].repeat(42, 1) for i in range(id)])
        ico_sphere_mesh = ico_sphere(level=1, device=self.device)
        verts, faces = ico_sphere_mesh.get_mesh_verts_faces(0)
        verts = verts * radius
        verts = verts.repeat(points.shape[0], 1, 1).to('cuda')
        faces = faces.repeat(points.shape[0], 1, 1).to('cuda')

        verts += points
        B, V, _ = verts.shape
        colors = torch.tensor([[0, 1, 0]]).to('cuda').unsqueeze(1).expand(B, V, -1)[..., :3]
        return verts, faces, colors

    def set_human_trajectory(self, verts_human, radius=0.015):
        points = verts_human[0][4292].repeat(42, 1)
        ico_sphere_mesh = ico_sphere(level=1, device=self.device)
        verts, faces = ico_sphere_mesh.get_mesh_verts_faces(0)
        verts = verts * radius
        verts = verts.to('cuda')
        faces = faces.to('cuda')

        verts += points
        V, _ = verts.shape
        colors = torch.tensor([0.8, 0.8, 0.8]).to('cuda').expand(V, -1)[..., :3]
        self.human_trajectory_verts.append(verts)
        self.human_trajectory_faces.append(faces)
        self.human_trajectory_colors.append(colors)

    def set_ground(self, length, center_x, center_z, offset=0.0):
        device = self.device
        v, f, vc, fc = map(torch.from_numpy, checkerboard_geometry(length=length, c1=center_x, c2=center_z, up="y", offset=offset))
        v, f, vc = v.to(device), f.to(device), vc.to(device)
        self.ground_geometry = [v, f, vc]

    def update_camera_params(self, focal, principt):
        """bbox in the format of xyxy"""
        
        self.K = torch.tensor(
            [[focal[0], 0, principt[0]],
            [0, focal[1], principt[1]],
            [0, 0, 1]]
        ).unsqueeze(0).float().to(self.device)

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
        self.cameras = self.create_camera()
        self.create_renderer()

    def update_bbox_from_points(self, x3d, scale=2.0, mask=None):
        """ Update bbox of cameras from the given 3d points

        x3d: input 3D keypoints (or vertices), (num_frames, num_points, 3)
        """
        if x3d.size(-1) != 3:
            x2d = x3d.unsqueeze(0)
        else:
            x2d = perspective_projection(x3d.unsqueeze(0), self.K, self.R, self.T.reshape(1, 3, 1))

        if mask is not None:
            x2d = x2d[:, ~mask]

        bbox = compute_bbox_from_points(x2d, self.width, self.height, scale)
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def reset_bbox(self,):
        bbox = torch.zeros((1, 4)).float().to(self.device)
        bbox[0, 2] = self.width
        bbox[0, 3] = self.height
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def render_mesh(self, focal, pcp_pt, vertices, background, colors=[0.8, 0.8, 0.8]):
        self.update_camera_params(focal, pcp_pt)
        vertices = vertices.unsqueeze(0)
        
        if colors[0] > 1: colors = [c / 255. for c in colors]
        verts_features = torch.tensor(colors).reshape(1, 1, 3).to(device=vertices.device, dtype=vertices.dtype)
        verts_features = verts_features.repeat(1, vertices.shape[1], 1)
        textures = TexturesVertex(verts_features=verts_features)
        
        mesh = Meshes(verts=vertices,
                      faces=self.faces,
                      textures=textures,)
        
        materials = Materials(
            device=self.device,
            specular_color=(colors, ),
            shininess=0
            )

        results = torch.flip(
            self.renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights),
            [1, 2]
        )

        image = results[0, ..., :3] * 255
        mask = results[0, ..., -1] > 1e-3

        image = overlay_image_onto_background(image, mask, self.bboxes, background.copy())
        self.reset_bbox()
        return image
    
    
    def render_with_ground(self, id, verts, faces, colors, cameras, lights):
        """
        :param verts (B, V, 3)
        :param faces (F, 3)
        :param colors (B, 3)
        """
        # (B, V, 3), (B, F, 3), (B, V, 3)
        verts, faces, colors = prep_shared_geometry(verts, faces, colors)
        # (V, 3), (F, 3), (V, 3)
        gv, gf, gc = self.ground_geometry
        cv, cf, cc = self.cam_geometry[id]
        ctv, ctf, ctc = self.set_cam_trajectory(id + 1)
        self.set_human_trajectory(verts)

        verts = list(torch.unbind(verts, dim=0)) + [gv] + [cv] + self.human_trajectory_verts + list(torch.unbind(ctv, dim=0))
        faces = list(torch.unbind(faces, dim=0)) + [gf] + [cf] + self.human_trajectory_faces + list(torch.unbind(ctf, dim=0))
        colors = list(torch.unbind(colors, dim=0)) + [gc[..., :3]] + [cc[..., :3]] + self.human_trajectory_colors + list(torch.unbind(ctc, dim=0))
        mesh = create_meshes(verts, faces, colors)

        materials = Materials(
            device=self.device,
            shininess=0
        )
        
        results = self.renderer(mesh, cameras=cameras, lights=lights, materials=materials)
        image = (results[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            
        return image
    
    
def prep_shared_geometry(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (F, 3)
    :param colors (B, 4)
    """
    B, V, _ = verts.shape
    F, _ = faces.shape
    colors = colors.unsqueeze(1).expand(B, V, -1)[..., :3]
    faces = faces.unsqueeze(0).expand(B, F, -1)
    return verts, faces, colors


def create_meshes(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (B, F, 3)
    :param colors (B, V, 3)
    """
    textures = TexturesVertex(verts_features=colors)
    meshes = Meshes(verts=verts, faces=faces, textures=textures)
    return join_meshes_as_scene(meshes)


def get_global_cameras(verts, device, distance=50, position=(-5.0, 5.0, 0.0)):
    positions = torch.tensor([position]).repeat(len(verts), 1)
    targets = verts.mean(1).mean(0)
    targets= torch.tensor(targets).repeat(len(verts), 1)
    
    directions = targets - positions
    directions = directions / torch.norm(directions, dim=-1).unsqueeze(-1) * distance
    positions = targets - directions
    
    rotation = look_at_rotation(positions, targets, ).mT
    translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)
    
    lights = PointLights(device=device, location=[position])
    return rotation, translation, lights


def _get_global_cameras(verts, device, min_distance=3, chunk_size=100):
    
    # split into smaller chunks to visualize
    start_idxs = list(range(0, len(verts), chunk_size))
    end_idxs = [min(start_idx + chunk_size, len(verts)) for start_idx in start_idxs]
    
    Rs, Ts = [], []
    for start_idx, end_idx in zip(start_idxs, end_idxs):
        vert = verts[start_idx:end_idx].clone()

def checkerboard_geometry(
    length=12.0,
    color0=[0.8, 0.9, 0.9],
    color1=[0.6, 0.7, 0.7], #[0.8, 0.9, 0.9],
    tile_width=0.5,
    alpha=1.0,
    up="y",
    c1=0.0,
    c2=0.0,
    offset=0.0
):
    assert up == "y" or up == "z"
    color0 = np.array(color0 + [alpha])
    color1 = np.array(color1 + [alpha])
    radius = length / 2.0
    num_rows = num_cols = max(2, int(length / tile_width))
    vertices = []
    vert_colors = []
    faces = []
    face_colors = []
    for i in range(num_rows):
        for j in range(num_cols):
            u0, v0 = j * tile_width - radius, i * tile_width - radius
            us = np.array([u0, u0, u0 + tile_width, u0 + tile_width])
            vs = np.array([v0, v0 + tile_width, v0 + tile_width, v0])
            zs = np.zeros(4)
            if up == "y":
                cur_verts = np.stack([us, zs, vs], axis=-1)  # (4, 3)
                cur_verts[:, 0] += c1
                cur_verts[:, 2] += c2
                cur_verts[:, 1] += offset
            else:
                cur_verts = np.stack([us, vs, zs], axis=-1)  # (4, 3)
                cur_verts[:, 0] += c1
                cur_verts[:, 1] += c2
                cur_verts[:, 2] += offset

            cur_faces = np.array(
                [[0, 1, 3], [1, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64
            )
            cur_faces += 4 * (i * num_cols + j)  # the number of previously added verts
            use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
            cur_color = color0 if use_color0 else color1
            cur_colors = np.array([cur_color, cur_color, cur_color, cur_color])

            vertices.append(cur_verts)
            faces.append(cur_faces)
            vert_colors.append(cur_colors)
            face_colors.append(cur_colors)

    vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    vert_colors = np.concatenate(vert_colors, axis=0).astype(np.float32)
    faces = np.concatenate(faces, axis=0).astype(np.float32)
    face_colors = np.concatenate(face_colors, axis=0).astype(np.float32)

    return vertices, faces, vert_colors, face_colors

def camera_marker_geometry(radius, height, up):
    assert up == "y" or up == "z"
    if up == "y":
        vertices = torch.tensor(
            [
                [-radius, -radius, 0],
                [radius, -radius, 0],
                [radius, radius, 0],
                [-radius, radius, 0],
                [0, 0, -height],
            ]
        , dtype=torch.float32)
    else:
        vertices = torch.tensor(
            [
                [-radius, 0, -radius],
                [radius, 0, -radius],
                [radius, 0, radius],
                [-radius, 0, radius],
                [0, -height, 0],
            ]
        , dtype=torch.float32)

    faces = torch.tensor(
        [[0, 3, 1], [1, 3, 2], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],]
    , dtype=torch.int32).to('cuda')

    face_colors = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    , dtype=torch.float32).to('cuda')
    return vertices, faces, face_colors