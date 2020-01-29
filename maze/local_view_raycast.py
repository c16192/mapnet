from math import pi, ceil, cos, sin
import torch
from .local_view import AgentView


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
