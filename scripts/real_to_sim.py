# # FROM https://github.com/simpler-env/SimplerEnv?tab=readme-ov-file#installation

# import simpler_env
# from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

# env = simpler_env.make('google_robot_pick_horizontal_coke_can')
# obs, reset_info = env.reset()
# instruction = env.get_language_instruction()
# print("Reset info", reset_info)
# print("Instruction", instruction)

# done, truncated = False, False
# while not (done or truncated):
#    # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
#    # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
#    image = get_image_from_maniskill2_obs_dict(env, obs)
#    action = env.action_space.sample() # replace this with your policy inference
#    obs, reward, done, truncated, info = env.step(action) # for long horizon tasks, you can call env.advance_to_next_subtask() to advance to the next subtask; the environment might also autoadvance if env._elapsed_steps is larger than a threshold
#    new_instruction = env.get_language_instruction()
#    if new_instruction != instruction:
#       # for long horizon tasks, we get a new instruction when robot proceeds to the next subtask
#       instruction = new_instruction
#       print("New Instruction", instruction)

# episode_stats = info.get('episode_stats', {})
# print("Episode stats", episode_stats)


import site
site.main()

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import mediapy
import sapien.core as sapien

task_name = "google_robot_pick_coke_can"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]

if 'env' in locals():
  print("Closing existing env")
  env.close()
  del env
env = simpler_env.make(task_name)
# Colab GPU does not supoort denoiser
sapien.render_config.rt_use_denoiser = False
obs, reset_info = env.reset()
instruction = env.get_language_instruction()
print("Reset info", reset_info)
print("Instruction", instruction)

frames = []
done, truncated = False, False
while not (done or truncated):
   # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
   # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
   image = get_image_from_maniskill2_obs_dict(env, obs)
   action = env.action_space.sample() # replace this with your policy inference
   obs, reward, done, truncated, info = env.step(action)
   frames.append(image)

episode_stats = info.get('episode_stats', {})
print("Episode stats", episode_stats)
mediapy.show_video(frames, fps=10)