from maze.loader import load_envs_from_txt_file
import os.path

base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "data", "maze")


def test_load_envs_from_txt_file():
    filename = os.path.join(base_dir, "mazes-10-10-100000.txt")
    map_size = (21, 21)
    envs = load_envs_from_txt_file(filename, map_size)
    print(envs[0])


test_load_envs_from_txt_file()