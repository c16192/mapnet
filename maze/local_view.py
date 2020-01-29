from math import pi, cos, sin, tan, ceil, floor, atan
from utils import rotate90
import torch


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
        :return:
        """
        center = pi / 2 * ang90
        return [(center - self.view_angle, center + self.view_angle)]


class AgentViewRayCast(AgentView):
    def __init__(self, view_range, no_rotation=False):
        super(AgentViewRayCast, self).__init__()
        self.rays_for_angles = get_rays(view_range, no_rotation)

    def local(self, square_patch, ang90):
        # project rays that position/orientation
        rays = self.rays_for_angles[ang90]
        h, w = square_patch.shape
        visible_patch = raycast(rays, square_patch, (h - 1) // 2, (w - 1) // 2)
        return visible_patch

    def glob(self, env, x, y, ang90):
        # project rays that position/orientation
        rays = self.rays_for_angles[ang90]
        image = raycast(rays, env, x, y)
        return image


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


def get_partially_observable_pixels(square_patch, angle_ranges, split_required=True):
    """
    :param square_patch: 0-1 array of shape (2k+1, 2k+1), where obstacle-filled pixels are Trues
    :param angle_ranges:
        if split_required == True:
            list of tuples of feasible ranges of view
            [(a1, b1), (a2, b2), ...] where (a1 < b1), (a2 < b2), ...
        else:
            list of tuples of feasible ranges of view [(a1, b1, s1), (a2, b2, s2),...]
            where a < b and (a, b) fits into (-pi/4, pi/4), (pi/4, 3pi/4), (3pi/4, 5pi/4), (5pi/4, 7pi/4)
            and s denotes the side (0: right, 1: up, 2: left, 3: bottom)
    :param split_required: set to true for initial call to convert angle_ranges into the right format
        for subsequent recurrent calls:
    :return:
        visible_patch: boolean array of shape (2, 2k+1, 2k+1)
            1st channel: True if they are visible clear pixels
            2nd channel: True if they are visible wall pixels
        angle_ranges: list of tuples of feasible ranges of view that are still not blocked
    """
    if split_required:
        angle_ranges = split_angle_ranges(angle_ranges)
    x, y = square_patch.shape
    assert x == y and x % 2 == 1
    # map of visible pixels to be returned
    visible_patch = torch.zeros(2, x, y, dtype=torch.long)
    if square_patch.shape == (1, 1):
        visible_patch[0, 0, 0] = 1
        return visible_patch, angle_ranges
    # call function recursively to get visible patch for the inner (2k-1, 2k-1) patch, and update the angle_ranges
    visible_patch[:, 1:-1, 1:-1], angle_ranges =\
        get_partially_observable_pixels(square_patch[1:-1, 1:-1], angle_ranges, split_required=False)

    visible_patch, angle_ranges = update_visiblility(square_patch, visible_patch, angle_ranges, x / 2 - 1, y)
    visible_patch, angle_ranges = update_visiblility(square_patch, visible_patch, angle_ranges, x / 2, y)
    return visible_patch, angle_ranges


def update_visiblility(square_patch, visible_patch, angle_ranges, r, y):
    new_angle_range = []

    for ang_sm, ang_lg, side in angle_ranges:
        # rotate so that ang_sm and ang_lg is in range (-pi/4, pi/4)
        square_patch = rotate90(square_patch, -side)
        visible_patch = rotate90(visible_patch, -side)
        rotate = (pi / 2) * side

        # update strip visibility
        square_patch[-1, :], visible_patch[:, -1, :], strip_ang_ranges\
            = eval_strip_visibility(square_patch[-1, :], visible_patch[:, -1, :], ang_sm - rotate, ang_lg - rotate, r, y)

        # rotate back to original angle
        new_angle_range += [(s + rotate, l + rotate, side) for s, l in strip_ang_ranges]
        square_patch = rotate90(square_patch, side)
        visible_patch = rotate90(visible_patch, side)
    return visible_patch, new_angle_range


def eval_strip_visibility(map_strip, visible_strip, ang_sm, ang_lg, r, y):
    new_angle_range = []
    new_ang_sm = None
    for i in range(
            max(0, floor(y / 2 + r * tan(ang_sm))),
            min(int(y / 2 + r), ceil(y / 2 + r * tan(ang_lg)))
    ):
        if not map_strip[i]:
            # visible clear spaces
            visible_strip[0, i] = 1
            if new_ang_sm is None:
                new_ang_sm = max(ang_sm, atan((i - y / 2) / r))
        else:
            # visible walls
            visible_strip[1, i] = 1
            if new_ang_sm is not None:
                new_arg_lg = min(ang_lg, atan((i - y / 2) / r))
                new_angle_range.append((new_ang_sm, new_arg_lg))
                new_ang_sm = None

    if new_ang_sm is not None:
        new_angle_range.append((new_ang_sm, ang_lg))

    return map_strip, visible_strip, new_angle_range


def split_angle_ranges(angle_ranges):
    """
    make sure the ranges of (ang_sm, ang_lg) fit into (-pi/4, pi/4), (pi/4, 3pi/4), (3pi/4, 5pi/4), (5pi/4, 7pi/4)
    where each corresponds to the bottom, right, top, and left edges respectively
    :param angle_ranges: list of tuples of feasible ranges of view
        [(a1, b1), (a2, b2), ...] where (a1 < b1), (a2 < b2), ...
    :return: list of tuples of feasible ranges of view [(a1, b1, s1), (a2, b2, s2),...]
         where a < b and (a, b) fits into (-pi/4, pi/4), (pi/4, 3pi/4), (3pi/4, 5pi/4), (5pi/4, 7pi/4)
         and s denotes the side (0: bottom, 1: right, 2: top, 3: left)
    """
    adjusted_ranges = []
    while angle_ranges:
        ang_s, ang_l = angle_ranges.pop(0)
        assert 0 <= ang_l - ang_s <= 2 * pi
        # rotate both ang_s and ang_l by pi/4
        ang_s += pi / 4
        ang_l += pi / 4
        quadrant = ang_s // (pi / 2)
        # if ang_s and ang_l don't share quadrants, divide the range so that they do
        if ang_l // (pi / 2) > quadrant:
            next_quadrant = (quadrant + 1) * (pi / 2)
            # rotate next_ang_s and next_ang_l by -pi/4
            angle_ranges.append((next_quadrant - pi / 4, ang_l - pi / 4))
            ang_l = next_quadrant
        # normalize angle to ranges (-pi/4, 7pi/4)
        adj_ang_s = ang_s % (2 * pi) - pi / 4
        adj_ang_l = ang_l + adj_ang_s - ang_s
        adjusted_ranges.append((adj_ang_s, adj_ang_l, int(quadrant % 4)))

    return adjusted_ranges


def get_rays(view_range, no_rotation=False):
    # raycast and store resulting lines, starting from the origin, for all 4 directions
    r = view_range  # radius
    perimeter = 2 * pi * r
    rays_for_angles = []
    for ang90 in range(1 if no_rotation else 4):
        if no_rotation:  # fixed, full 360 deg view
            angles = torch.linspace(0, 2 * pi, int(ceil(perimeter)))
        else:  # oriented, 180 FOV
            base_angle = ang90 * pi / 2
            angles = torch.linspace(base_angle - pi / 2, base_angle + pi / 2, int(ceil(perimeter)))

        rays = [tuple(bresenham_line(0, 0, round(r * cos(a)), round(r * sin(a)))) for a in angles]
        rays = list(set(rays))  # remove duplicates
        # rays = [t.tensor(r, dtype=t.long) for r in rays]
        rays_for_angles.append(rays)
    return rays_for_angles


# also see: https://github.com/ActiveState/code/blob/3b27230f418b714bc9a0f897cb8ea189c3515e99/recipes/Python/578112_Bresenhams_line_algorithm/recipe-578112.py
def bresenham_line(x, y, x2, y2):
    """Brensenham line algorithm"""
    steep = 0
    coords = []
    sx = 1 if (x2 - x) > 0 else -1
    sy = 1 if (y2 - y) > 0 else -1
    dx = abs(x2 - x)
    dy = abs(y2 - y)
    if dy > dx:
        steep = 1
        x, y = y, x
        dx, dy = dy, dx
        sx, sy = sy, sx
    d = (2 * dy) - dx
    for i in range(0, dx):
        if steep:
            coords.append((y, x))
        else:
            coords.append((x, y))
        while d >= 0:
            y = y + sy
            d = d - (2 * dx)
        x = x + sx
        d = d + (2 * dy)
    coords.append((x2, y2))
    return coords


def raycast(rays, env, x, y):
    """Raycast in an environment to black-out any non-visible tiles"""
    image = torch.zeros((2, env.shape[0], env.shape[1]))
    for ray in rays:
        # assert isinstance(ray, tuple)
        for (rx, ry) in ray:
            v = int(env[x + rx, y + ry].item())
            image[v, x + rx, y + ry] = 1
            if v: break  # hit a wall
    return image
