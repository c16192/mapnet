from maze.local_view import bresenham_line, get_rays, raycast, split_angle_ranges, \
    get_partially_observable_pixels, embed_view, AgentViewAnyAngles
import torch
from math import pi


def get_env1(map_size):
    env = torch.ones((map_size, map_size), dtype=torch.long)
    env[:, 2] = False
    env[2, :] = False
    env[0, 3] = False
    return env


def get_env2(map_size):
    env = torch.ones((map_size, map_size), dtype=torch.long)
    env[:, 2] = False
    env[2, :] = False
    env[1, 3] = False
    env[0, 4] = False
    return env


def get_env3():
    map_size = 7
    env = torch.ones((map_size, map_size), dtype=torch.long)
    env[:, 3] = False
    env[3, :] = False
    env[0, 4] = False
    env[1, 2] = False
    return env


def test_bresenham_line():
    print(bresenham_line(0, 0, 1, 2))


def test_raycast():
    r = view_range = 2  # radius
    map_size = r * 2 + 1
    env1 = get_env1(map_size)
    print("Obstacles in env1")
    print(env1)
    env2 = get_env2(map_size)
    print("Obstacles in env2")
    print(env2)

    rays = get_rays(view_range, no_rotation=True)
    image1 = raycast(rays[0], env1, 2, 2)

    print("After raycast for Env 1")
    print(image1)

    image2 = raycast(rays, env2, 2, 2, 0)
    print("After raycast for Env 2")
    print(image2)


def test_split_angle_range():
    angs = split_angle_ranges([(-3, 1)])
    print(angs)


def get_env():
    map_size = 7
    env = (torch.rand((map_size, map_size)) > 0.5).int()
    return env


def test_get_partially_observable_pixels():
    env = get_env()
    print(env)
    visible, angle_range = get_partially_observable_pixels(env, [(0, 2*pi)])
    print(visible, angle_range)


def test_embed_view():
    patch = torch.stack((get_env3(), get_env3()))
    print(patch)
    env = embed_view(patch, (7, 7), 1, 2, 2)
    print(env)


def test_agent_view_any_angles():
    env = get_env3()
    print(env)
    agent_view = AgentViewAnyAngles(view_range=2, view_angle=pi)
    glob_view = agent_view.glob(env, 4, 3, 1)
    print(glob_view)


# test_split_angle_range()
test_get_partially_observable_pixels()
test_embed_view()
test_agent_view_any_angles()