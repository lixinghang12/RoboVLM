      
import json
import os
import random
import math
import warnings
import traceback
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from typing import List, Callable, Literal, Union, Dict
import tqdm
from utils.model_utils import build_tokenizer
from data.data_utils import get_text_function, get_prompt_builder
from model.policy_head.action_tokenizer import ActionTokenizer

# def alpha2rotm(a):
#     """Alpha euler angle to rotation matrix."""
#     rotm = np.array([
#         [1, 0, 0],
#         [0, np.cos(a), -np.sin(a)],
#         [0, np.sin(a),  np.cos(a)]
#     ])
#     return rotm

def alpha2rotm(a):
    """Alpha euler angle to rotation matrix.
    """
    if isinstance(a, float):
        rotm = np.array([
            [1, 0, 0],
            [0, np.cos(a), -np.sin(a)],
            [0, np.sin(a),  np.cos(a)]
        ])
        return rotm
    elif isinstance(a, list) or isinstance(a, np.ndarray):
        rotm = np.zeros((a.shape[0], 3, 3))
        rotm[:, 0, 0] = 1
        rotm[:, 1, 1] = np.cos(a)
        rotm[:, 1, 2] = -np.sin(a)
        rotm[:, 2, 1] = np.sin(a)
        rotm[:, 2, 2] = np.cos(a)
        return rotm
    else:
        raise TypeError("a must be float, list or ndarray")

def beta2rotm(b):
    """Beta euler angle to rotation matrix."""
    if isinstance(b, float):
        rotm = np.array([
            [np.cos(b), 0, np.sin(b)],
            [0, 1, 0],
            [-np.sin(b), 0, np.cos(b)]
        ])
        return rotm
    elif isinstance(b, list) or isinstance(b, np.ndarray):
        b = np.array(b)
        rotm = np.zeros((b.shape[0], 3, 3))
        rotm[:, 0, 0] = np.cos(b)
        rotm[:, 0, 2] = np.sin(b)
        rotm[:, 1, 1] = 1
        rotm[:, 2, 0] = -np.sin(b)
        rotm[:, 2, 2] = np.cos(b)
        return rotm
    else:
        raise TypeError("b must be float, list or ndarray")


def gamma2rotm(c):
    """Gamma euler angle to rotation matrix."""
    if isinstance(c, float):
        rotm = np.array([
            [np.cos(c), -np.sin(c), 0],
            [np.sin(c),  np.cos(c), 0],
            [0, 0, 1]
        ])
        return rotm
    elif isinstance(c, list) or isinstance(c, np.ndarray):    
        rotm = np.zeros((c.shape[0], 3, 3))
        rotm[:, 0, 0] = np.cos(c)
        rotm[:, 0, 1] = -np.sin(c)
        rotm[:, 1, 0] = np.sin(c)
        rotm[:, 1, 1] = np.cos(c)
        rotm[:, 2, 2] = 1
        return rotm
    else:
        raise TypeError("c must be float, list or ndarray")

def euler2rotm(euler_angles):
    """Euler angle (ZYX) to rotation matrix."""
    alpha = euler_angles[..., 0]
    beta = euler_angles[..., 1]
    gamma = euler_angles[..., 2]

    rotm_a = alpha2rotm(alpha)
    rotm_b = beta2rotm(beta)
    rotm_c = gamma2rotm(gamma)

    rotm = rotm_c @ rotm_b @ rotm_a

    return rotm


def isRotm(R: np.ndarray):
    # Checks if a matrix is a valid rotation matrix.
    # Forked from Andy Zeng
    if len(R.shape) == 2:
        Rt = np.transpose(R)
    elif len(R.shape) == 3:
        Rt = np.transpose(R, (0, 2, 1))
    else:
        raise TypeError("R must be a 2D or 3D array")
    
    shouldBeIdentity = Rt @ R
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotm2euler(R: np.ndarray):
    # Forked from: https://learnopencv.com/rotation-matrix-to-euler-angles/
    # R = Rz * Ry * Rx
    assert(isRotm(R))
    # sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    # singular = sy < 1e-6
 
    # if not singular :
    #     x = math.atan2(R[2,1] , R[2,2])
    #     y = math.atan2(-R[2,0], sy)
    #     z = math.atan2(R[1,0], R[0,0])
    # else:
    #     x = math.atan2(-R[1,2], R[1,1])
    #     y = math.atan2(-R[2,0], sy)
    #     z = 0
    
    # # (-pi , pi]
    # while x > np.pi:
    #     x -= (2 * np.pi)
    # while x <= -np.pi:
    #     x += (2 * np.pi)
    # while y > np.pi:
    #     y -= (2 * np.pi)
    # while y <= -np.pi:
    #     y += (2 * np.pi)
    # while z > np.pi:
    #     z -= (2 * np.pi)
    # while z <= -np.pi:
    #     z += (2 * np.pi)
    # return np.array([x, y, z])

def rotm2euler(Rs):
    assert Rs.shape[-2:] == (3, 3), "Input must be of shape (b, 3, 3)"
    
    sy = np.sqrt(Rs[..., 0, 0]**2 + Rs[..., 1, 0]**2)
    singular = sy < 1e-6
    
    x = np.zeros_like(sy)
    y = np.zeros_like(sy)
    z = np.zeros_like(sy)
    
    x[~singular] = np.arctan2(Rs[~singular, 2, 1], Rs[~singular, 2, 2])
    y[~singular] = np.arctan2(-Rs[~singular, 2, 0], sy[~singular])
    z[~singular] = np.arctan2(Rs[~singular, 1, 0], Rs[~singular, 0, 0])
    
    x[singular] = np.arctan2(-Rs[singular, 1, 2], Rs[singular, 1, 1])
    y[singular] = np.arctan2(-Rs[singular, 2, 0], sy[singular])
    
    # Normalize angles to (-pi, pi]
    x = np.mod(x + np.pi, 2*np.pi) - np.pi
    y = np.mod(y + np.pi, 2*np.pi) - np.pi
    z = np.mod(z + np.pi, 2*np.pi) - np.pi
    
    return np.stack([x, y, z], axis=-1)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

    def forward_traj(self, x):
        n, t, c, h, w = x.size()
        x = x.view(n*t, *x.shape[2:])
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        base_grid = base_grid.unsqueeze(1).repeat(1, t, 1, 1, 1)
        base_grid = base_grid.view(n*t, *base_grid.shape[2:])
        shift = torch.randint(1,
                              2 * self.pad + 1,
                              size=(n*t, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        x = x.view(n, t, *x.shape[1:])
        return x


class Real_Dataset(Dataset):
    def __init__(
        self,
        anns_dir: str,
        media_dir: str,
        is_training: bool=True,
        action_type: str='rel_actions',
        c_act_scaler: List[int]=[100, 100, 100, 50, 50, 50],
        skip_frames: int=1,
        **kwargs
    ):
        """Constructor."""
        super().__init__()
        self.skip_frames = skip_frames
        self.action_type = action_type

        self.mode = 'train' if is_training else 'val'
        self.anns_dir = os.path.join(anns_dir, self.mode)
        self.media_dir = media_dir

        self.c_act_scaler = c_act_scaler
        if isinstance(c_act_scaler, (int, float)):
            self.c_act_scaler = [self.c_act_scaler] * 6
        self.c_act_scaler = np.array(self.c_act_scaler, dtype=float)

        self.action_dim = 7  # ee xyz (3) + ee euler (3) + gripper(1)
        self.state_dim = 7  # ee xyz (3) + ee euler (3) + gripper(1)

        self.ann_files = self._init_anns(self.anns_dir)
        # TODO delete the code for debug
        # self.ann_files = self.ann_files[:100]
        self.samples = self._get_samples(self.ann_files)
        self.samples = np.concatenate(self.samples, axis=0)
        save_name = f'{self.action_type}-{self.anns_dir.split("/")[-2]}.npy'
        np.save(save_name, self.samples)

        
    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.anns_dir}"

    def _init_anns(self, data_dir):
        ann_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
        return ann_files

    def _get_samples(self, ann_files):
        samples = []
        for ann_file in tqdm.tqdm(ann_files):
            with open(ann_file, "r") as f:
                ann = json.load(f)
            n_frames = len(ann['state'])
            # Add start and end
            start = ann['videos'][0]['start']
            end = ann['videos'][0]['end']
            if len(ann['videos']) > 1:
                for i in range(1, len(ann['videos'])):
                    assert ann['videos'][i]['start'] == start
                    assert ann['videos'][i]['end'] == end
            assert (end - start) == n_frames
            frame_ids = np.arange(0, n_frames, self.skip_frames)
            states = np.array(ann['state'])
            states = states[frame_ids]
            states = np.concatenate(
                [states, [ann['state'][-1]]],
                axis=0
            )
            actions = self._get_action_sequence(states, self.action_type)
            # if actions[:, 0].max() > 1:
            samples.append(actions)
        return samples

    def __len__(self):
        return len(self.samples)

    def _get_action_sequence(self, states: np.ndarray, action_type: Literal["rel_actions", "relative"]):
        """
        Args:
            states (np.ndarray): shape (n_frames, 7)
        """
        if action_type == "rel_actions":
            states = np.array(states)
            actions = states[1:] - states[:-1]
            actions[:, -1] = states[1:, -1]
        elif action_type == 'relative':
            first_xyz = states[:-1, :3]
            first_rpy = states[:-1, 3:6]
            first_rotm = euler2rotm(first_rpy)
            next_xyz = states[1:, :3]
            next_rpy = states[1:, 3:6]
            next_rotm = euler2rotm(next_rpy)
            
            rel_rotm = first_rotm.transpose(0, 2, 1) @ next_rotm
            rel_rpy = rotm2euler(rel_rotm)
            rel_xyz = next_xyz - first_xyz
            rel_xyz = first_rotm.transpose(0, 2, 1) @ rel_xyz[..., None]
            rel_xyz = rel_xyz[..., 0]
            actions = np.concatenate([rel_xyz, rel_rpy, states[1:, -1:]], axis=-1)
        
        actions[..., -1] = (actions[..., -1] + 1) / 2
        return actions

    def __getitem__(self, index):
        if self.mode == 'val':
            np.random.seed(index)
            random.seed(index)
        sample = self.samples[index]
        return sample

def count(file_name: str):
    data = np.load(file_name)
    xyz = data[:, :3].reshape(-1)
    rpy = data[:, 3:6].reshape(-1)

    xyz = np.sort(xyz)
    rpy = np.sort(rpy)

    min_01_data_index = int(len(xyz) * 0.01)
    max_99_data_index = int(len(xyz) * 0.99)

    min_01_xyz = xyz[min_01_data_index]
    min_01_rpy = rpy[min_01_data_index]

    max_99_xyz = xyz[max_99_data_index]
    max_99_rpy = rpy[max_99_data_index]
    
    print(f"xyz: {min_01_xyz}, {max_99_xyz}")
    print(f"rpy: {min_01_rpy}, {max_99_rpy}")
    with open('.cache/real_data/scale.json', 'r') as file:
        data = json.load(file)

    data[file_name.split('.')[0]] = (min_01_xyz, max_99_xyz, min_01_rpy, max_99_rpy)
    with open('.cache/real_data/scale.json', 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    
    seq_len=8
    # debug
    # anns_dir = "/mnt/bn/robotics-data-lxh-lq/RoboVLM/debug_dataset/anns/0606_medium"
    # media_dir = "/mnt/bn/robotics-data-lxh-lq/RoboVLM/debug_dataset/media/0524_medium"

    # medium
    # anns_dir =  "/mnt/bn/robotics-data-lxh-lq/real_data_medium/anns/0606_medium"
    # media_dir = "/mnt/bn/robotics-data-lxh-lq/real_data_medium/media/0524_medium"

    anns_dir =  "/mnt/bn/robotics-data-lxh-lq/real_data/anns/0419_easy"
    media_dir = "/mnt/bn/robotics-data-lxh-lq/real_data/media/0419_easy"

    dataset = Real_Dataset(
        anns_dir=anns_dir,
        media_dir=media_dir,
        is_training=True,
        c_act_scaler=[100, 100, 100, 50.0, 50.0, 50.0],
        # action_type="relative"
    )
