import os

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender

from calvin_env.envs.play_table_env import get_env

path = "/mnt/bn/robotics/manipulation_data/calvin_data/task_ABCD_D/validation"
env = get_env(path, show_gui=False)
print(env.get_obs())