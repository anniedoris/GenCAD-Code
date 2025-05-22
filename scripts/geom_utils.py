import numpy as np
import math

## The code in this file is taken from the DeepCAD repo

class CoordSystem(object):
    """
    Local coordinate system for sketch plane.
    Code taken from the DeepCAD repo.
    """
    def __init__(self, origin, theta, phi, gamma, y_axis=None, is_numerical=False):
        self.origin = origin
        self._theta = theta # 0~pi
        self._phi = phi     # -pi~pi
        self._gamma = gamma # -pi~pi
        self._y_axis = y_axis # (theta, phi)
        self.is_numerical = is_numerical

    @property
    def normal(self):
        return polar2cartesian([self._theta, self._phi])

    @property
    def x_axis(self):
        normal_3d, x_axis_3d = polar_parameterization_inverse(self._theta, self._phi, self._gamma)
        return x_axis_3d

    @property
    def y_axis(self):
        if self._y_axis is None:
            return np.cross(self.normal, self.x_axis)
        return polar2cartesian(self._y_axis)

    @staticmethod
    def from_dict(stat):
        origin = np.array([stat["origin"]["x"], stat["origin"]["y"], stat["origin"]["z"]])
        normal_3d = np.array([stat["z_axis"]["x"], stat["z_axis"]["y"], stat["z_axis"]["z"]])
        x_axis_3d = np.array([stat["x_axis"]["x"], stat["x_axis"]["y"], stat["x_axis"]["z"]])
        y_axis_3d = np.array([stat["y_axis"]["x"], stat["y_axis"]["y"], stat["y_axis"]["z"]])
        theta, phi, gamma = polar_parameterization(normal_3d, x_axis_3d)
        return CoordSystem(origin, theta, phi, gamma, y_axis=cartesian2polar(y_axis_3d))

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        origin = vec[:3]
        theta, phi, gamma = vec[3:]
        system = CoordSystem(origin, theta, phi, gamma)
        if is_numerical:
            system.denumericalize(n)
        return system

    def __str__(self):
        return "origin: {}, normal: {}, x_axis: {}, y_axis: {}".format(
            self.origin.round(4), self.normal.round(4), self.x_axis.round(4), self.y_axis.round(4))

    def transform(self, translation, scale):
        self.origin = (self.origin + translation) * scale

    def numericalize(self, n=256):
        """NOTE: shall only be called after normalization"""
        # assert np.max(self.origin) <= 1.0 and np.min(self.origin) >= -1.0 # TODO: origin can be out-of-bound!
        self.origin = ((self.origin + 1.0) / 2 * n).round().clip(min=0, max=n-1).astype(np.int)
        tmp = np.array([self._theta, self._phi, self._gamma])
        self._theta, self._phi, self._gamma = ((tmp / np.pi + 1.0) / 2 * n).round().clip(
            min=0, max=n-1).astype(np.int)
        self.is_numerical = True

    def denumericalize(self, n=256):
        self.origin = self.origin / n * 2 - 1.0
        tmp = np.array([self._theta, self._phi, self._gamma])
        self._theta, self._phi, self._gamma = (tmp / n * 2 - 1.0) * np.pi
        self.is_numerical = False

    def to_vector(self):
        return np.array([*self.origin, self._theta, self._phi, self._gamma])


def rads_to_degs(rads):
    """Convert an angle from radians to degrees"""
    return 180 * rads / math.pi


def angle_from_vector_to_x(vec):
    """computer the angle (0~2pi) between a unit vector and positive x axis"""
    angle = 0.0
    # 2 | 1
    # -------
    # 3 | 4
    if vec[0] >= 0:
        if vec[1] >= 0:
            # Qadrant 1
            angle = math.asin(vec[1])
        else:
            # Qadrant 4
            angle = 2.0 * math.pi - math.asin(-vec[1])
    else:
        if vec[1] >= 0:
            # Qadrant 2
            angle = math.pi - math.asin(vec[1])
        else:
            # Qadrant 3
            angle = math.pi + math.asin(-vec[1])
    return angle


def cartesian2polar(vec, with_radius=False):
    """convert a vector in cartesian coordinates to polar(spherical) coordinates"""
    vec = vec.round(6)
    norm = np.linalg.norm(vec)
    theta = np.arccos(vec[2] / norm) # (0, pi)
    phi = np.arctan(vec[1] / (vec[0] + 1e-15)) # (-pi, pi) # FIXME: -0.0 cannot be identified here
    if not with_radius:
        return np.array([theta, phi])
    else:
        return np.array([theta, phi, norm])


def polar2cartesian(vec):
    """convert a vector in polar(spherical) coordinates to cartesian coordinates"""
    r = 1 if len(vec) == 2 else vec[2]
    theta, phi = vec[0], vec[1]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def rotate_by_x(vec, theta):
    mat = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    return np.dot(mat, vec)


def rotate_by_y(vec, theta):
    mat = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(mat, vec)


def rotate_by_z(vec, phi):
    mat = np.array([[np.cos(phi), -np.sin(phi), 0],
                    [np.sin(phi), np.cos(phi), 0],
                    [0, 0, 1]])
    return np.dot(mat, vec)


def polar_parameterization(normal_3d, x_axis_3d):
    """represent a coordinate system by its rotation from the standard 3D coordinate system

    Args:
        normal_3d (np.array): unit vector for normal direction (z-axis)
        x_axis_3d (np.array): unit vector for x-axis

    Returns:
        theta, phi, gamma: axis-angle rotation 
    """
    normal_polar = cartesian2polar(normal_3d)
    theta = normal_polar[0]
    phi = normal_polar[1]

    ref_x = rotate_by_z(rotate_by_y(np.array([1, 0, 0]), theta), phi)

    gamma = np.arccos(np.dot(x_axis_3d, ref_x).round(6))
    if np.dot(np.cross(ref_x, x_axis_3d), normal_3d) < 0:
        gamma = -gamma
    return theta, phi, gamma


def polar_parameterization_inverse(theta, phi, gamma):
    """build a coordinate system by the given rotation from the standard 3D coordinate system"""
    normal_3d = polar2cartesian([theta, phi])
    ref_x = rotate_by_z(rotate_by_y(np.array([1, 0, 0]), theta), phi)
    ref_y = np.cross(normal_3d, ref_x)
    x_axis_3d = ref_x * np.cos(gamma) + ref_y * np.sin(gamma)
    return normal_3d, x_axis_3d


def get_arc(x, y, curr_x, curr_y, alpha, flag, is_numerical=True):
    end_point = np.array([x, y])
    start_point = np.array([curr_x, curr_y])
    sweep_angle = np.array(alpha) / 256 * 2 * np.pi if is_numerical else vec[3]
    clock_sign = np.array(flag)
    s2e_vec = end_point - start_point
    if np.linalg.norm(s2e_vec) == 0:
        return None
    radius = (np.linalg.norm(s2e_vec) / 2) / np.sin(sweep_angle / 2)
    s2e_mid = (start_point + end_point) / 2
    vertical = np.cross(s2e_vec, [0, 0, 1])[:2]
    vertical = vertical / np.linalg.norm(vertical)
    if clock_sign == 0:
        vertical = -vertical
    center_point = s2e_mid - vertical * (radius * np.cos(sweep_angle / 2))

    start_angle = 0
    end_angle = sweep_angle
    if clock_sign == 0:
        ref_vec = end_point - center_point
    else:
        ref_vec = start_point - center_point
    ref_vec = ref_vec / np.linalg.norm(ref_vec)
    
    # Get the midangle
    mid_angle = (start_angle + end_angle) / 2
    rot_mat = np.array([[np.cos(mid_angle), -np.sin(mid_angle)],
                        [np.sin(mid_angle), np.cos(mid_angle)]])
    mid_vec = rot_mat @ ref_vec
    mid_point =  center_point + mid_vec * radius

    return start_point, mid_point, end_point