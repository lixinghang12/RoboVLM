import numpy as np

TRIAL_IDX = 0
INPUT_SIZE = 192


TEXT_I = 9
TEXTS = {
    0: ("pick up the toast bread from the red bowl", True),
    1: ("place the picked object into the toaster", True),
    2: ("press the toaster switch", False),
    3: ("pick up the toast bread from the toaster", True),
    4: ("place the picked object in the red bowl", True),
    5: ("pick up the white mug from the tray", True),
    6: ("place the picked object under the coffee spout", True),
    7: ("press the coffee machine button", False),
    8: ("pick up the white mug from under the coffee spout", True),
    9: ("place the picked object on the tray", True),
}

TEXT, GRIPPER_CHANGE_TERMINATE = TEXTS[TEXT_I]
INIT_POS = [0.527, -0.403, 0.926]
INIT_QUAT = [0.943, 0.172, -0.272, 0.086]


# Experiment
SAVE_TEST_IMAGE = False
RANDOM_RANGE = [0.02, 0.02, 0.02]
USE_RANDOM = True

MAX_ITERS = 500
# SERVER_IP = "10.79.157.179"
SERVER_IP = "192.168.1.11"
PORT = 10243

# Gripper
GRIPPER_THRESHOLD = 10
GRIPPER_OPEN = 1
GRIPPER_CLOSE = 0

# Mode 3
WORLD_OFFSET_ROTM = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
WORLD_OFFSET_POS = np.array([0.0, 0.0, 0.0])
HAND_OFFSET_ROTM = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
HAND_OFFSET_POS = np.array([0.0, 0.0, 0.0])
CROP_DIMS = [[50, 200], [690, 1000]]
ACTION_SCALE = [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
