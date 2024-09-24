import torch

import os
import time

print(
    "|| MASTER_ADDR:",os.environ["MASTER_ADDR"],
    "|| MASTER_PORT:",os.environ["MASTER_PORT"],
    "|| LOCAL_RANK:",os.environ["LOCAL_RANK"],
    "|| RANK:",os.environ["RANK"], 
    "|| WORLD_SIZE:",os.environ["WORLD_SIZE"]
    )
