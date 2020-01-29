import numpy as np
import torch


def load_envs_from_txt_file(filename, env_size):
    """
    Reads a text file that contains "#" for walls and " " for clear space, returning a map as a boolean array
    :param filename: file path to the dataset
    :param env_size: tuple of (map_height, map_width)
    :return: a boolean PyTorch tensor of shape (n_examples, map_height, map_width), where True corresponds to walls
    """
    # read all text at once
    with open(filename, 'r') as f:
        text = f.read()

    # convert to a 1D tensor of bytes
    data = torch.tensor(list(text.encode()), dtype=torch.uint8)

    # reshape into a stack of lines
    data = data.reshape(-1, env_size[0] + 1)

    # remove line break at the end of each line
    assert (data[:, -1] == ord('\n')).all()
    data = data[:, :-1]

    # reshape to split lines into list of environments along first dim
    envs = (data.reshape(-1, env_size[0], env_size[1]) == ord('#')).int()
    return envs
