from random import randint, seed
from time import time

import torch
from torch.utils.data.dataset import Dataset
from maze.local_view import AgentViewAnyAngles, extract_view
import os
from math import pi
from maze.loader import load_envs_from_txt_file


class Mazes(Dataset):
    """Dataset of 2D mazes, simulating local view with occlusions. The view is returned as a Bx2xHxW byte tensor."""

    def __init__(self, envs, view_range=5, seq_length=5, clear_threshold=3, max_speed=0):

        self.envs = envs
        self.view_range = view_range
        self.seq_length = seq_length
        self.clear_threshold = clear_threshold
        self.max_speed = max_speed
        self.agent_view = AgentViewAnyAngles(view_range)

    def __len__(self):
        """Return number of environments"""
        return self.envs.shape[0]

    def __getitem__(self, index):
        """Return a random sequence of images from the environment with given index"""
        start = time()
        # get environment
        env = self.envs[index, :, :]

        # list of possible initial positions [(h0, w0), (h1, w1),...]: free space (where elements of env are zeros)
        pos = (1 - env).nonzero()

        # output tensor with image sequence
        images = torch.zeros((self.seq_length, 2, self.view_range * 2 + 1, self.view_range * 2 + 1), dtype=torch.uint8)
        poses = []
        frame = 0

        for tries in range(10 * self.seq_length):
            # choose random position (restricted to available choices)
            h, w = pos[randint(0, pos.shape[0] - 1), :].tolist()
            assert env[h, w] == 0  # sanity check, tile shouldn't be a wall

            # choose random orientation
            rot90 = randint(0, 3)

            image = self.agent_view.glob(env, h, w, rot90)

            # accept it if there are enough visible clear tiles
            visible_clear = image[0, :, :]
            if visible_clear.sum() >= self.clear_threshold:
                # jump to a visible position next
                new_pos = visible_clear.nonzero()

                # avoid this position if there are no valid tiles reachable within the max speed (distance from one frame to the next)
                if self.max_speed:
                    is_close = ((new_pos - torch.tensor([[h, w]])) ** 2).sum(dim=1) <= self.max_speed ** 2
                    if is_close.sum() == 0:
                        continue
                    new_pos = new_pos[is_close, :]

                pos = new_pos

                # store results for this frame
                images[frame, :, :, :] = extract_view(image, h, w, 2-rot90, self.view_range)
                poses.append((h, w, pi / 2 * rot90))

                frame += 1
                if frame >= self.seq_length: break  # done

        poses = torch.tensor(poses, dtype=torch.float)
        return {'images': images, 'poses': poses, 'time': time() - start}


if __name__ == '__main__':
    from overboard import tshow

    seed(0)  # repeatable random sequence

    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "data", "maze", "mazes-10-10-100000.txt")
    envs = load_envs_from_txt_file(filepath, env_size=(21, 21))
    mazes = Mazes(envs)

    tshow(mazes.envs[0:6, :, :])

    images = [mazes[0]['images'] for _ in range(10)]
    images = torch.stack(images, dim=0).int()
    images = images[:, :, 0, :, :] - images[:, :, 1, :, :]  # difference between wall and ground

    tshow(images)
    input()
