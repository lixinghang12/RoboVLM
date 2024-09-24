      
from functools import partial
import json
import os
from pathlib import Path
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
from data.data_utils import get_text_function, get_prompt_builder, preprocess_image
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
        image_fn: Callable,
        tokenizer: Dict,
        anns_dir: str,
        media_dir: str,
        discrete_action: bool=False,
        action_tokenizer=None,
        model_name: str="vicuna",
        predict_stop_token: bool=True,
        window_size: int=10,
        fwd_pred_next_n: int=1,
        is_training: bool=True,
        action_type: str='rel_actions',
        c_act_scaler: List[int]=[100, 100, 100, 50, 50, 50],
        skip_frames: int=1,
        rgb_pad: int=-1,
        gripper_pad: int=-1,
        traj_cons: bool=True,
        cache_root: str='.cache/real_data',
        **kwargs
    ):
        """Constructor."""
        super().__init__()
        self.skip_frames = skip_frames
        self.window_size = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.image_fn = image_fn
        # self.tokenizer = tokenizer
        self.cache_root = cache_root
        assert isinstance(tokenizer, dict)
        tokenizer_type = tokenizer['tokenizer_type']
        max_text_len = tokenizer['max_text_len']
        tokenizer = build_tokenizer(tokenizer_config=tokenizer)
        self.tokenizer = tokenizer
        self.text_fn = get_text_function(tokenizer, tokenizer_type, max_text_len)
        self.action_type = action_type
        self.model_name = model_name
        self.discrete_action = discrete_action
        self.predict_stop_token = predict_stop_token
        if self.discrete_action:
            self.action_tokenizer = action_tokenizer or ActionTokenizer(self.tokenizer)
        else:
            self.action_tokenizer = None

        self.rgb_shift = RandomShiftsAug(rgb_pad) if rgb_pad != -1 else None
        self.gripper_shift = RandomShiftsAug(gripper_pad) if gripper_pad != -1 else None
        self.traj_cons = traj_cons
        self.mode = 'train' if is_training else 'val'
        self.anns_dir = os.path.join(anns_dir, self.mode)
        self.media_dir = media_dir

        self.c_act_scaler = c_act_scaler
        if isinstance(c_act_scaler, (int, float)):
            self.c_act_scaler = [self.c_act_scaler] * 6
        self.c_act_scaler = np.array(self.c_act_scaler, dtype=float)
        self.ann_files = self._init_anns(self.anns_dir)
        self.discrete_scale = self.get_discrete_scale() if self.discrete_action else None

        self.samples = self._get_samples_with_cache(self.ann_files)

        self.action_dim = 7  # ee xyz (3) + ee euler (3) + gripper(1)
        self.state_dim = 7  # ee xyz (3) + ee euler (3) + gripper(1)

        
    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.anns_dir}"

    def _init_anns(self, data_dir):
        ann_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
        return ann_files

    def _get_samples_with_cache(self, ann_files):
        dataset_id = self.get_dataset_id()
        cache_file_path = Path(self.cache_root)
        cache_file_path = cache_file_path / dataset_id
        if self.discrete_action:
            cache_file_path = cache_file_path / 'discrete'
        else:
            cache_file_path = cache_file_path / 'continuous'
        c_act_scaler_str = '-'.join([str(_) for _ in self.c_act_scaler.astype(int).tolist()])
        cache_file_path = cache_file_path / f'{self.mode}_{self.window_size}_{self.fwd_pred_next_n}_{self.skip_frames}_{c_act_scaler_str}.json'

        if os.path.exists(cache_file_path):
            with open(cache_file_path, 'r') as f:
                samples = json.load(f)
        else:
            samples = self._get_samples(ann_files)
            # os.makedirs(cache_file_path.parent, exist_ok=True)
            # with open(cache_file_path, 'w') as f:
            #     json.dump(samples, f)
        return samples
    
    def _get_samples(self, ann_files):
        samples = []
        for ann_file in tqdm.tqdm(ann_files, desc="collate samples"):
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
            if len(frame_ids) < self.window_size + self.fwd_pred_next_n:
                continue

            states = np.array(ann['state'])
            states = states[frame_ids]
            states = np.concatenate(
                [states, [ann['state'][-1]]],
                axis=0
            )
            actions = self._get_action_sequence(states, self.action_type)
            window_size = self.window_size + self.fwd_pred_next_n
            sample_num = len(frame_ids) - window_size + 1

            for i in range(sample_num):
                sample = dict()
                sample['ann_file'] = ann_file
                sample['video_frame_ids'] = (frame_ids[i: i+window_size] + start).tolist()
                sample['actions'] = actions[i:i+window_size].tolist()
                sample['states'] = states[i:i+window_size].tolist()
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def transform_action_by_disrete_scale(self, actions: np.ndarray):
        def linear_map(x, a, b):
            # linear map [a, b] to [-1, 1]
            return (2.0 / (b - a)) * (x - a) - 1

        min_xyz, max_xyz, min_rpy, max_rpy = self.discrete_scale
        actions[..., :3] = linear_map(actions[..., :3], min_xyz, max_xyz)
        actions[..., 3:6] = linear_map(actions[..., 3:6], min_rpy, max_rpy)
        actions = np.clip(actions, -1, 1)
        return actions

    def _get_action_sequence(self, states: np.ndarray, action_type: Literal["rel_actions", "relative"], raw: bool=False):
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
        if not raw:
            if not self.discrete_action:
                actions[..., :6] *= self.c_act_scaler
            else:
                actions = self.transform_action_by_disrete_scale(actions)
        return actions

    def _load_video_decord(self, video_path, frame_ids):
        """Load video content using Decord"""
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        return frame_data
    
    def _get_frames(self, video_info, frame_ids):
        video_path = video_info['video_path']
        remain_path="/".join(video_path.split("/")[-3:])
        video_path=os.path.join(self.media_dir, remain_path)
        crop = video_info.get('crop', None)

        frames = self._load_video_decord(video_path, frame_ids)
        if crop is not None:
            frames = frames[:, crop[0][0]:crop[1][0], crop[0][1]:crop[1][1]]
        frames = frames.astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        # make frames to tensor
        frames = self.image_fn(frames)
        return frames

    def _get_obs(self, label, frame_ids):
        obs = [None, None]
        videos = label['videos']
        for video_i, video in enumerate(videos):
            assert video_i in [0, 1]
            if video_i == 0:
                frames = self._get_frames(video, frame_ids)
            elif video_i == 1:
                frames = self._get_frames(video, frame_ids)
            obs[video_i] = frames
        return obs[0], obs[1]

    def __getitem__(self, index):
        if self.mode == 'val':
            np.random.seed(index)
            random.seed(index)
        sample = self.samples[index]
        ann_file = sample['ann_file']
        video_frame_ids = sample['video_frame_ids']
        with open(ann_file, "r") as f:
            label = json.load(f)
        static_rgbs, hand_rgbs = self._get_obs(label, video_frame_ids)
        return {
            'text': label['texts'][0],
            'static_rgbs': static_rgbs,
            'hand_rgbs': hand_rgbs,
            'actions': sample['actions'],
            'states': sample['states']
        }

    def transform_image_tensor(self, shift: Union[RandomShiftsAug, None], image_tensors: torch.Tensor):
        if shift is None:
            return image_tensors
            
        bs, seq_len = image_tensors.shape[:2]
        if self.traj_cons:
            image_tensors = shift.forward_traj(image_tensors)
        else:
            image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
            image_tensors = shift(image_tensors)
            image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
        return image_tensors
    
    def wrap_instruction_and_action(self, lang, action):
        # modified from OpenVLA
        IGNORE_INDEX = -100
        prompt_builder = get_prompt_builder(self.model_name, eos=self.tokenizer.eos_token, bos=self.tokenizer.bos_token)
        # if pass in multi-step actions, we concat them
        action = action.flatten()
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"\
             if self.window_size == 1 else f"What {self.window_size} step actions should the robot take to {lang}?"},
            {"from": "gpt", "value": ""},
        ]
        input_ids = []
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        action[..., -1] = action[..., -1] * 2 - 1  # convert [0, 1] to [-1, 1]
        action_ids = self.action_tokenizer.encode_actions_to_token_ids(action)
        if self.tokenizer.eos_token is None:
            input_ids = input_ids + action_ids
        else:
            input_ids = input_ids[:-1] + action_ids + [input_ids[-1]]
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        if self.tokenizer.eos_token is None:
            labels[: -len(action)] = IGNORE_INDEX
        else:        
            labels[: -(len(action) + 1)] = IGNORE_INDEX
        
        if not self.predict_stop_token and self.tokenizer.eos_token:
            labels[-1] = IGNORE_INDEX

        return input_ids, labels
    
    def collater(self, samples):
        action_tensors = torch.from_numpy(np.stack([s['actions'] for s in samples]))
        state_tensors = torch.from_numpy(np.stack([s['states'] for s in samples]))
        image_tensors = torch.stack([s['static_rgbs'] for s in samples])
        gripper_tensors = torch.stack([s['hand_rgbs'] for s in samples])
        image_tensors = self.transform_image_tensor(self.rgb_shift, image_tensors)
        gripper_tensors = self.transform_image_tensor(self.gripper_shift, gripper_tensors)
        text_tensors, text_attention_mask = self.text_fn([s['text'] for s in samples])
        action_tensors = action_tensors[:, :-1]

        action_tensors = action_tensors.to(image_tensors.dtype)
        state_tensors = state_tensors.to(image_tensors.dtype)

        if self.fwd_pred_next_n != 1:
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.fwd_pred_next_n, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.fwd_pred_next_n]

            states = torch.zeros((action_tensors.shape[0], self.window_size, self.fwd_pred_next_n, state_tensors.shape[-1]))
            for b in range(state_tensors.shape[0]):
                for ix in range(self.window_size):
                    states[b, ix] = state_tensors[b, ix:ix+self.fwd_pred_next_n]
            states = torch.cat([states[..., :6], states[..., -1:]], dim=-1)

            action_tensors = actions
            state_tensors = states

        image_tensors = image_tensors[:, :-self.fwd_pred_next_n]
        gripper_tensors = gripper_tensors[:, :-self.fwd_pred_next_n]
        
        bs = len(samples)
        instr_and_action_ids = None
        instr_and_action_labels = None
        instr_and_action_mask = None
        
        if self.discrete_action:
            res = [self.wrap_instruction_and_action(s["text"], np.array(s["actions"])) for s in samples]
            tmp_input_ids = [_[0] for _ in res]
            tmp_labels = [_[1] for _ in res]
            
            max_len = max([len(_) for _ in tmp_input_ids])
            instr_and_action_ids = torch.zeros((bs, max_len), dtype=torch.long)
            instr_and_action_labels = torch.ones((bs, max_len), dtype=torch.long) * -100
            instr_and_action_mask = torch.zeros((bs, max_len), dtype=torch.bool)
            
            for i in range(bs):
                instr_and_action_ids[i, :len(tmp_input_ids[i])] = tmp_input_ids[i]
                instr_and_action_labels[i, :len(tmp_labels[i])] = tmp_labels[i]
                instr_and_action_mask[i, :len(tmp_input_ids[i])] = 1

        return {
            "rgb": image_tensors,
            "hand_rgb": gripper_tensors,
            "action": action_tensors,
            "text": text_tensors,
            "text_mask": text_attention_mask,
            "fwd_rgb_chunck": image_tensors.clone().unsqueeze(2),
            "fwd_hand_rgb_chunck": gripper_tensors.clone().unsqueeze(2),
            "action_chunck": action_tensors.clone().unsqueeze(2),
            "instr_and_action_ids": instr_and_action_ids,
            "instr_and_action_labels": instr_and_action_labels,
            "instr_and_action_mask": instr_and_action_mask,
            "data_source": "real_action"
        }

    def get_dataset_id(self):
        return f'{self.action_type}-{self.anns_dir.split("/")[-2]}'

    def get_raw_actions_cache_file_path(self):
        cache_name = "raw_actions.npy"
        return os.path.join(self.cache_root, self.get_dataset_id(), cache_name)

    def get_raw_actions(self) -> Union[np.ndarray,None]:
        cache_file = self.get_raw_actions_cache_file_path()
        if not os.path.exists(cache_file):
            return None
        return np.load(cache_file)
    
    def save_raw_actions(self, data: np.ndarray):
        cache_file = self.get_raw_actions_cache_file_path()
        cache_file = Path(cache_file)
        os.makedirs(cache_file.parent, exist_ok=True)
        np.save(str(cache_file), data)

    def get_scale_file_path(self):
        cache_name = 'scale.json'
        return os.path.join(self.cache_root, cache_name)

    def get_scale_data(self):
        scale_file = self.get_scale_file_path()
        if not os.path.exists(scale_file):
            return {}
        with open(scale_file) as file:
            data = json.load(file)
        return data
    
    def save_scale_data(self, data: Dict):
        scale_file = self.get_scale_file_path()
        scale_file = Path(scale_file)
        os.makedirs(scale_file.parent, exist_ok=True)
        with open(scale_file, 'w') as file:
            json.dump(data, file)

    def get_training_ann_files(self):
        if self.mode == 'train':
            return self.ann_files
        else:
            anns_dir = self.anns_dir.replace('val', 'train')
            ann_files = self._init_anns(anns_dir)
            return ann_files

    def _get_raw_actions(self):
        all_actions = []
        training_ann_files = self.get_training_ann_files()
        for ann_file in tqdm.tqdm(training_ann_files, desc="get raw actions"):
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
            actions = self._get_action_sequence(states, self.action_type, True)
            # if actions[:, 0].max() > 1:
            all_actions.append(actions)
        all_actions = np.concatenate(all_actions, axis=0)

        return all_actions
  
    def get_scale_from_raw_actions(self, raw_actions):
        xyz = raw_actions[:, :3].reshape(-1)
        rpy = raw_actions[:, 3:6].reshape(-1)

        xyz = np.sort(xyz)
        rpy = np.sort(rpy)

        min_01_data_index = int(len(xyz) * 0.01)
        max_99_data_index = int(len(xyz) * 0.99)

        min_01_xyz = xyz[min_01_data_index]
        min_01_rpy = rpy[min_01_data_index]

        max_99_xyz = xyz[max_99_data_index]
        max_99_rpy = rpy[max_99_data_index]
        return (min_01_xyz, max_99_xyz, min_01_rpy, max_99_rpy)

    def get_discrete_scale(self):
        scale_data = self.get_scale_data()
        dataset_id = self.get_dataset_id()
        if dataset_id in scale_data:
            return scale_data[dataset_id]
        
        raw_actions = self.get_raw_actions()
        if raw_actions is None:
            raw_actions = self._get_raw_actions()
            self.save_raw_actions(raw_actions)
        
        scale_data[dataset_id] = self.get_scale_from_raw_actions(raw_actions)
        self.save_scale_data(scale_data)
        return scale_data[dataset_id]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    
    seq_len=8
    # data_dir = "/mnt/bn/robotics-data-wht-yg/lpy/real_data/anns/0419_easy"
    # anns_dir = "/mnt/bn/robotics-data-lxh-lq/RoboVLM/debug_dataset/anns/0606_medium"
    # media_dir = "/mnt/bn/robotics-data-lxh-lq/RoboVLM/debug_dataset/media/0524_medium"
    
    # anns_dir =  "/mnt/bn/robotics-data-lxh-lq/real_data/anns/0419_easy"
    # media_dir = "/mnt/bn/robotics-data-lxh-lq/real_data/media/0419_easy"
    
    anns_dir =  "/mnt/bn/robotics-data-lxh-lq/real_data_medium/anns/0606_medium"
    media_dir = "/mnt/bn/robotics-data-lxh-lq/real_data_medium/media/0524_medium"

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    preprocess = transforms.Compose([
        transforms.Resize(
            (448, 448),
            interpolation=InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    tokenizer_config = {
        "type": "AutoTokenizer",
        "pretrained_model_name_or_path": "/mnt/bn/robotics-data-lxh-lq/lxh/Qwen-VL",
        "tokenizer_type": "qwen",
        "max_text_len": 64,
        "use_img_start_end": False,
        "image_start_token": "<im_start>",
        "image_end_token": "<im_end>",
        "use_act_start_end": False,
        "action_start_token": "<act_start>",
        "action_end_token": "<act_end>",
        "use_bbox_start_end": False,
        "bbox_start_token": "<bbox_start>",
        "bbox_end_token": "<bbox_end>"
    }
    image_fn = partial(preprocess_image, image_processor=preprocess)
    dataset = Real_Dataset(
        image_fn=image_fn,
        tokenizer=tokenizer_config,
        anns_dir=anns_dir,
        media_dir=media_dir,
        discrete_action=True,
        model_name='qwen',
        window_size=2,
        fwd_pred_next_n=3,
        is_training=True,
        action_type='relative',
        skip_frames=1,
        c_act_scaler=[100, 100, 100, 50.0, 50.0, 50.0],
    )
    dataset.collater([dataset[0], dataset[1]])
    from IPython import embed; embed()
