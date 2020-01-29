#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shu Ishida   University of Oxford
#
# Distributed under terms of the MIT license.

from math import pi, tan, ceil, floor, atan
from utils import rotate90
import torch


def get_partially_observable_pixels(square_patch, angle_ranges, split_required=True):
    """
    :param square_patch: 0-1 array of shape (2k+1, 2k+1), where obstacle-filled pixels are 1s
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
