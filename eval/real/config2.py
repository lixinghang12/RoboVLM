import numpy as np

TRIAL_IDX = 0

TEXT = "close the drawer"

INIT_POS = [0.427, -0.353, 0.926]
INIT_QUAT = [0.943, 0.172, -0.272, 0.086]
GRIPPER_CHANGE_TERMINATE = False

# Experiment
SAVE_TEST_IMAGE = False
RANDOM_RANGE = [0.02, 0.02, 0.02]
USE_RANDOM = True

MAX_ITERS = 500
# SERVER_IP = "10.79.157.179"
SERVER_IP = "192.168.1.11"
PORT = 10241

# Gripper
GRIPPER_THRESHOLD = 25
GRIPPER_OPEN = 1
GRIPPER_CLOSE = 0

# Mode 3
WORLD_OFFSET_ROTM = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
WORLD_OFFSET_POS = np.array([0.0, 0.0, 0.0])
HAND_OFFSET_ROTM = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
HAND_OFFSET_POS = np.array([0.0, 0.0, 0.0])
CROP_DIMS = [[60, 110], [690, 950]]
ACTION_SCALE = [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
