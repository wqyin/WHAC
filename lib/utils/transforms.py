import torch

def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def invert_transformation(matrix):
    # Assuming matrix is a 4x4 PyTorch tensor representing rotation and translation
    R = matrix[:3, :3]  # Extract the top-left 3x3 rotation matrix
    t = matrix[:3, 3]   # Extract the translation vector
    
    # Compute the inverse rotation by transposing R
    R_inv = R.t()
    
    # Compute the inverse translation
    t_inv = -torch.matmul(R_inv, t)
    
    # Construct the inverse transformation matrix
    inverse_matrix = torch.eye(4)  # Start with an identity matrix
    inverse_matrix[:3, :3] = R_inv
    inverse_matrix[:3, 3] = t_inv
    
    return inverse_matrix

def quat_to_rotmat(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
