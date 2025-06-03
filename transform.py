from collections import OrderedDict
from functools import reduce, partial

import mitsuba as mi
import numpy as np

from utils.registry import Registry

TRANSFORM = Registry("transforms")


def build_transform(cfg):
    return TRANSFORM.build(cfg=cfg)


@TRANSFORM.register_module()
class Translate:
    def __init__(self, translate, scalar: bool = True):
        super().__init__()
        self.vec = translate
        self.transform = mi.ScalarTransform4f if scalar else mi.Transform4f

    def func(self):
        return self.transform.translate(self.vec)

    def __call__(self, data):
        data[:, :3] += self.vec
        return data


@TRANSFORM.register_module()
class Scale:
    def __init__(self, scale, scalar: bool = True):
        super().__init__()
        self.vec = scale
        self.transform = mi.ScalarTransform4f if scalar else mi.Transform4f

    def func(self):
        return self.transform.scale(self.vec)

    def __call__(self, data):
        return data


@TRANSFORM.register_module()
class LookAt:
    def __init__(self, origin, target, up, scalar: bool = True):
        super().__init__()
        self.origin = origin
        self.target = target
        self.up = up
        self.transform = mi.ScalarTransform4f if scalar else mi.Transform4f

    def func(self):
        return self.transform.look_at(self.origin, self.target, self.up)


@TRANSFORM.register_module()
class ChainTransform:
    def __init__(self, transforms: list):
        self.trans = OrderedDict()
        for i, tran in enumerate(transforms):
            self.trans[tran["type"]] = build_transform(tran)

    def func(self):
        func = reduce(lambda x, y: x.func() @ y.func(), self.trans.values())
        return func

    def __call__(self, data):
        for name, tran in self.trans.items():
            data = tran(data)
        return data

    def update(self, tran: TRANSFORM):
        self.trans[tran["type"]] = build_transform(tran)


@TRANSFORM.register_module()
class Normalize:
    def __init__(self, scale_range=1.0):
        self.scale_range = scale_range

    def __call__(self, data):
        cord = data[:, :3]
        min_bound = np.min(cord, axis=0)
        max_bound = np.max(cord, axis=0)
        center = (min_bound + max_bound) / 2
        scale = np.max(max_bound - min_bound)
        data[:, :3] = (cord - center) / scale * self.scale_range

        return data


@TRANSFORM.register_module()
class DownSample:
    def __init__(self, n_sample, algorithm="random"):
        self.n_sample = n_sample
        if algorithm == "random":
            self.algorithm = partial(self.random, n_sample=self.n_sample)
        elif algorithm == "fps":
            self.algorithm = partial(self.fps, n_sample=self.n_sample)
        else:
            raise ValueError(f"Algorithm {algorithm} not supported")

    def __call__(self, data):
        index = self.algorithm(data[:, :3])
        return data[index]

    @staticmethod
    def random(data, n_sample):
        replace = n_sample > data.shape[0]
        indices_to_keep = np.random.choice(
            data.shape[0], size=n_sample, replace=replace
        )

        return indices_to_keep

    @staticmethod
    def fps(data, n_sample, start_idx=None):
        # farthest point sampling
        """Farthest Point Sampling without the need to compute all pairs of distance.

        Parameters
        ----------
        arr : numpy array
            The positional array of shape (n_points, n_dim)
        n_sample : int
            The number of points to sample.
        start_idx : int, optional
            If given, appoint the index of the starting point,
            otherwise randomly select a point as the start point.
            (default: None)

        Returns
        -------
        numpy array of shape (n_sample,)
            The sampled indices.

        Examples
        --------
        >>> import numpy as np
        >>> data = np.random.rand(100, 1024)
        >>> point_idx = farthest_point_sampling(data, 3)
        >>> print(point_idx)
            array([80, 79, 27])

        >>> point_idx = farthest_point_sampling(data, 5, 60)
        >>> print(point_idx)
            array([60, 39, 59, 21, 73])
        """
        n_points, n_dim = data.shape

        if (start_idx is None) or (start_idx < 0):
            start_idx = np.random.randint(0, n_points)

        sampled_indices = [start_idx]
        min_distances = np.full(n_points, np.inf)

        for _ in range(n_sample - 1):
            current_point = data[sampled_indices[-1]]
            dist_to_current_point = np.linalg.norm(data - current_point, axis=1)
            min_distances = np.minimum(min_distances, dist_to_current_point)
            farthest_point_idx = np.argmax(min_distances)
            sampled_indices.append(farthest_point_idx)

        return np.array(sampled_indices)


@TRANSFORM.register_module()
class ToNumpy:
    def __call__(self, data):
        if isinstance(data, np.ndarray):
            return data
        return np.array(data)


@TRANSFORM.register_module()
class Flip:
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, data):
        if self.axis == "x":
            data[:, 0] = -data[:, 0]
        elif self.axis == "y":
            data[:, 1] = -data[:, 1]
        elif self.axis == "z":
            data[:, 2] = -data[:, 2]
        else:
            raise ValueError(f"Axis {self.axis} not supported")
        return data


@TRANSFORM.register_module()
class Rotation:
    def __init__(self, axes, angles):
        if isinstance(axes, str):
            axes = list(axes)
        if not isinstance(angles, list):
            angles = [angles]

        if len(axes) != len(angles):
            raise ValueError("The number of axes must match the number of angles.")
        self.axes = axes
        self.angles = angles

    def __call__(self, data):
        min_bound = np.min(data[:, :3], axis=0)
        max_bound = np.max(data[:, :3], axis=0)
        center = (min_bound + max_bound) / 2

        # Apply the rotations in sequence
        for axis, angle in zip(self.axes, self.angles):
            rot_t = self.get_rotation_matrix(axis, angle)
            data[:, :3] = (data[:, :3] - center) @ rot_t.T + center

        return data

    def get_rotation_matrix(self, axis, angle):
        cos, sin = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        if axis == "x":
            return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])
        elif axis == "y":
            return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
        elif axis == "z":
            return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
        else:
            raise ValueError(f"Axis {axis} not supported")


@TRANSFORM.register_module()
class Colorizer:
    def __init__(self, color=None, **kwargs):
        self.color = color
        self.kwargs = kwargs

    def __call__(self, data):
        if self.color == 'rainbow':
            return self.rainbow_color(data, **self.kwargs)
        if self.color == 'gray':
            return self.gray_color(data, **self.kwargs)
        else:
            return data

    def rainbow_color(self, data, **kwargs):
        cord = data[:, :3]
        center = (np.max(cord, axis=0) + np.min(cord, axis=0)) / 2
        cord = center - cord + np.array([0.5, 0.5, 0.5])
        cord_clip = np.clip(cord, 0.001, 1.0)
        norm = np.sqrt(np.sum(cord_clip**2, axis=1, keepdims=True))
        color = cord_clip / norm
        return np.concatenate([data[:, :3], color], axis=1)

    def gray_color(self, data, **kwargs):
        brightness = kwargs.get('brightness', 0.5)
        color = np.ones_like(data[:, :3]) * brightness
        return np.concatenate([data[:, :3], color], axis=1)

@TRANSFORM.register_module()
class LiftUp:
    def __init__(self, gap: float):
        self.gap = gap

    def __call__(self, data):
        bottom = np.min(data[:, 2])
        data[:, 2] += self.gap - bottom
        return data

@TRANSFORM.register_module()
class AdjustSaturation:
    def __init__(self, saturation=1.0):
        self.saturation = saturation

    def __call__(self, data):
        color = data[:, 3:]
        gray = np.sum(color * np.array([0.2989, 0.5870, 0.1140]), axis=1, keepdims=True)

        color_adjusted = gray + self.saturation * (color - gray)
        data[:, 3:] = color_adjusted
        return data

try:
    import open3d as o3d
except ImportError:
    pass


@TRANSFORM.register_module()
class OcclusionEstimation:
    def __init__(self, view_pose, factor=5):
        self.view_pose = view_pose
        self.factor = factor

    def __call__(self, data):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(data[:, :3])
        diameter = np.linalg.norm(
            np.asarray(pcd_o3d.get_max_bound()) - np.asarray(pcd_o3d.get_min_bound())
        )
        radius = diameter * self.factor
        _, select_idx = pcd_o3d.hidden_point_removal(self.view_pose, radius)
        data = data[select_idx]
        return data
