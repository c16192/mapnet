from math import pi
from utils import rotate90
import torch
from .local_view_any_angles import get_partially_observable_pixels


class AgentView:
    def __init__(self):
        pass

    def local(self, square_patch, ang90):
        """
        :param square_patch: 0-1 array of shape (2k+1, 2k+1), where obstacle-filled pixels are Trues
        :param ang90: (0: bottom, 1: right, 2: top, 3: left)
        :return: visible_patch
        """
        raise NotImplementedError

    def glob(self, env, x, y, ang90):
        raise NotImplementedError


class AgentViewAnyAngles(AgentView):
    def __init__(self, view_range, view_angle=pi/2):
        super(AgentViewAnyAngles, self).__init__()
        self.view_range = view_range
        self.view_angle = view_angle

    def local(self, square_patch, ang90):
        angle_ranges = self.get_angle_ranges(ang90)
        visible_patch, _ = get_partially_observable_pixels(square_patch, angle_ranges)
        return visible_patch

    def glob(self, env, x, y, ang90):
        r = self.view_range
        angle_ranges = self.get_angle_ranges(ang90)
        square_patch = extract_view(env, x, y, 0, r)
        visible_patch, _ = get_partially_observable_pixels(square_patch, angle_ranges)
        return embed_view(visible_patch, env.shape, 0, x - r, y - r)

    def get_angle_ranges(self, ang90):
        """
        :param ang90: (0: bottom, 1: right, 2: top, 3: left)
        :return: angle range
        """
        center = pi / 2 * ang90
        return [(center - self.view_angle, center + self.view_angle)]


def extract_view(env, x, y, ang90, view_range):
    """Extract a local view from an environment at the given pose"""
    # get coordinates of window to extract
    xs = torch.arange(x - view_range, x + view_range + 1, dtype=torch.long)
    ys = torch.arange(y - view_range, y + view_range + 1, dtype=torch.long)

    # get coordinate 0 instead of going out of bounds
    (h, w) = env.shape[-2:]
    (invalid_xs, invalid_ys) = ((xs < 0) | (xs >= h), (ys < 0) | (ys >= w))  # coords outside the env
    xs[invalid_xs] = 0
    ys[invalid_ys] = 0

    # extract view, and set to 0 observations that were out of bounds
    # view = env[..., xs, ys]  # not equivalent to view = env[..., y1:y2, x1:x2]
    view = env.index_select(dim=-2, index=xs).index_select(dim=-1, index=ys)

    view[..., invalid_xs, :] = 0
    view[..., :, invalid_ys] = 0

    # rotate. note only 90 degrees rotations are allowed
    return rotate90(view, ang90)


def embed_view(patch, env_shape, ang90, x_start, y_start):
    """Embed a local view in an environment at the given pose"""
    patch = rotate90(patch, ang90)

    assert len(env_shape) == 2
    image = torch.zeros((*patch.shape[:-2], *env_shape), dtype=patch.dtype)
    env_h, env_w = env_shape
    patch_h, patch_w = patch.shape[-2:]
    image[..., max(0, x_start):patch_h + x_start, max(0, y_start):patch_w + y_start]\
        = patch[..., max(0, -x_start):env_h - x_start, max(0, -y_start):env_w - y_start]

    return image

