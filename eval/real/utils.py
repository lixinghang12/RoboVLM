import math
import numpy as np

def aa2rotm(angle, axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis_magnitude = np.linalg.norm(axis)
    axis = np.divide(axis,
                     axis_magnitude,
                     out=np.zeros_like(axis),
                     where=axis_magnitude != 0)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.array(np.outer(axis, axis) * (1.0 - cosa))
    axis *= sina
    RA = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                   [-axis[1], axis[0], 0.0]])
    R = RA + np.array(R)
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:

        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def isRotm(R):
    # Checks if a matrix is a valid rotation matrix.
    # Forked from Andy Zeng
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotm2aa(R):
    # From: euclideanspace.com

    epsilon = 0.01  # Margin to allow for rounding errors
    epsilon2 = 0.1  # Margin to distinguish between 0 and 180 degrees

    assert (isRotm(R))

    if ((abs(R[0][1] - R[1][0]) < epsilon)
            and (abs(R[0][2] - R[2][0]) < epsilon)
            and (abs(R[1][2] - R[2][1]) < epsilon)):
        # Singularity found
        # First check for identity matrix which must have +1 for all terms in leading diagonaland zero in other terms
        if ((abs(R[0][1] + R[1][0]) < epsilon2)
                and (abs(R[0][2] + R[2][0]) < epsilon2)
                and (abs(R[1][2] + R[2][1]) < epsilon2)
                and (abs(R[0][0] + R[1][1] + R[2][2] - 3) < epsilon2)):
            # this singularity is identity matrix so angle = 0
            return [0, 1, 0, 0]  # zero angle, arbitrary axis

        # Otherwise this singularity is angle = 180
        angle = np.pi
        xx = (R[0][0] + 1) / 2
        yy = (R[1][1] + 1) / 2
        zz = (R[2][2] + 1) / 2
        xy = (R[0][1] + R[1][0]) / 4
        xz = (R[0][2] + R[2][0]) / 4
        yz = (R[1][2] + R[2][1]) / 4
        if ((xx > yy) and (xx > zz)):  # R[0][0] is the largest diagonal term
            if (xx < epsilon):
                x = 0
                y = 0.7071
                z = 0.7071
            else:
                x = np.sqrt(xx)
                y = xy / x
                z = xz / x
        elif (yy > zz):  # R[1][1] is the largest diagonal term
            if (yy < epsilon):
                x = 0.7071
                y = 0
                z = 0.7071
            else:
                y = np.sqrt(yy)
                x = xy / y
                z = yz / y
        else:  # R[2][2] is the largest diagonal term so base result on this
            if (zz < epsilon):
                x = 0.7071
                y = 0.7071
                z = 0
            else:
                z = np.sqrt(zz)
                x = xz / z
                y = yz / z
        return [angle, x, y, z]  # Return 180 deg rotation

    # As we have reached here there are no singularities so we can handle normally
    s = np.sqrt((R[2][1] - R[1][2]) * (R[2][1] - R[1][2]) +
                (R[0][2] - R[2][0]) * (R[0][2] - R[2][0]) +
                (R[1][0] - R[0][1]) * (R[1][0] - R[0][1]))  # used to normalise
    if (abs(s) < 0.001):
        s = 1

    # Prevent divide by zero, should not happen if matrix is orthogonal and should be
    # Caught by singularity test above, but I've left it in just in case
    angle = np.arccos(np.clip((R[0][0] + R[1][1] + R[2][2] - 1) / 2, -1, 1))
    x = (R[2][1] - R[1][2]) / s
    y = (R[0][2] - R[2][0]) / s
    z = (R[1][0] - R[0][1]) / s
    return [angle, x, y, z]

def get_mat_log(R):
  """Get the log(R) of the rotation matrix R.
  
  Args:
    R (3x3 numpy array): rotation matrix
  Returns:
    w (3, numpy array): log(R)
  """
  theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
  w_hat = (R - R.T) * theta / (2 * np.sin(theta))  # Skew symmetric matrix
  w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])  # [w1, w2, w3]

  return w

def rotm2quat(R):
    """Get the quaternion from rotation matrix.
    
    Args:
        R (3x3 numpy array): rotation matrix
    Return:
        q (4, numpy array): quaternion, x, y, z, w
    """
    w = get_mat_log(R)
    theta = np.linalg.norm(w)

    if theta < 0.001:
        q = np.array([0, 0, 0, 1])
        return q

    axis = w / theta

    q = np.sin(theta/2) * axis
    q = np.r_[q, np.cos(theta/2)]

    return q

def rotm2quat_ros(R):
    from scipy.spatial.transform import Rotation as Rot
    rot = Rot.from_matrix(R)
    q = rot.as_quat()
    return q

def quat2rotm(quat):
    """Quaternion to rotation matrix."""
    w = quat[3]
    x = quat[0]
    y = quat[1]
    z = quat[2]
    s = w * w + x * x + y * y + z * z
    rotm = np.array(
        [
            [
                1 - 2 * (y * y + z * z) / s,
                2 * (x * y - z * w) / s,
                2 * (x * z + y * w) / s,
            ],
            [
                2 * (x * y + z * w) / s,
                1 - 2 * (x * x + z * z) / s,
                2 * (y * z - x * w) / s,
            ],
            [
                2 * (x * z - y * w) / s,
                2 * (y * z + x * w) / s,
                1 - 2 * (x * x + y * y) / s,
            ],
        ]
    )
    return rotm

def alpha2rotm(a):
    """Alpha euler angle to rotation matrix."""
    rotm = np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ])
    return rotm

def beta2rotm(b):
    """Beta euler angle to rotation matrix."""
    rotm = np.array([
        [np.cos(b), 0, np.sin(b)],
        [0, 1, 0],
        [-np.sin(b), 0, np.cos(b)]
    ])
    return rotm

def gamma2rotm(c):
    """Gamma euler angle to rotation matrix."""
    rotm = np.array([
        [np.cos(c), -np.sin(c), 0],
        [np.sin(c),  np.cos(c), 0],
        [0, 0, 1]
    ])
    return rotm

def euler2rotm(euler_angles):
    """Euler angle (ZYX) to rotation matrix."""
    alpha = euler_angles[0]
    beta = euler_angles[1]
    gamma = euler_angles[2]

    rotm_a = alpha2rotm(alpha)
    rotm_b = beta2rotm(beta)
    rotm_c = gamma2rotm(gamma)

    rotm = rotm_c @ rotm_b @ rotm_a

    return rotm

def skew_symmetric(axis):
    mat = np.zeros((3, 3))
    mat[0, 1] = -axis[2]
    mat[0, 2] =  axis[1]
    mat[1, 0] =  axis[2]
    mat[1, 2] = -axis[0]
    mat[2, 0] = -axis[1]
    mat[2, 1] =  axis[0]
    return mat

def angle2rotm(angle, axis):
    # Normalize axis
    axis = axis / np.linalg.norm(axis)
    axis_hat = skew_symmetric(axis)
    # Rodrigues' formula
    R = np.eye(3) + axis_hat * np.sin(angle) + (1 - np.cos(angle)) * axis_hat @ axis_hat
    return R

def rotm2euler(R):
    # Forked from: https://learnopencv.com/rotation-matrix-to-euler-angles/
    # R = Rz * Ry * Rx
    assert isRotm(R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    if x < 0:
        x += 2 * np.pi
    return np.array([x, y, z])
