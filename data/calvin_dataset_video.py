import functools
import io
import json
import logging
import os
import random
import tarfile
from dataclasses import dataclass
from multiprocessing import Value
from utils.model_utils import build_tokenizer
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from omegaconf import DictConfig
import pyhash
from typing import Any, Dict, Tuple, Callable
from itertools import chain
from calvin_agent.datasets.utils.episode_utils import lookup_naming_pattern
import pickle
import torch.nn as nn
import torch.nn.functional as F
from data.data_utils import get_prompt_builder, get_text_function, mu_law_companding, normalize_action, regularize_action
from model.policy_head.action_tokenizer import ActionTokenizer
from calvin_agent.datasets.utils.episode_utils import (
    get_state_info_dict,
    process_actions,
    process_depth,
    process_state,
)

Image.MAX_IMAGE_PIXELS = 1000000000
MAX_NUM_TOKENS = 256
MAX_NUM_IMAGES = 5
TINY_IMAGE_SIZE_THRESHOLD = 1
N_CHANNELS = 3
INTERLEAVED_IMAGE_SIZE = 224

MIN_KB = 10
MAX_NUM_IMAGES = 5

import logging
from pathlib import Path
from typing import Dict, Tuple, Union, Literal

hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)

obs_config = DictConfig(
    {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
)

prop_state = DictConfig(
    {
        "n_state_obs": 15,
        "keep_indices": [[0, 15]],
        "robot_orientation_idx": [3, 6],
        "normalize": True,
        "normalize_robot_orientation": True,
    }
)


def get_validation_window_size(
    idx: int, min_window_size: int, max_window_size: int
) -> int:
    """
    In validation step, use hash function instead of random sampling for consistent window sizes across epochs.

    Args:
        idx: Sequence index.
        min_window_size: Minimum window size.
        max_window_size: Maximum window size.

    Returns:
        Window size computed with hash function.
    """
    window_range = max_window_size - min_window_size + 1
    return min_window_size + hasher(str(idx)) % window_range


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
        x = x.reshape(n*t, *x.shape[2:])
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
        base_grid = base_grid.reshape(n*t, *base_grid.shape[2:])
        shift = torch.randint(1,
                              2 * self.pad + 1,
                              size=(n*t, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        x = x.reshape(n, t, *x.shape[1:])
        return x


class BaseCalvinDataset(Dataset):
    """
    Abstract dataset base class.

    Args:
        datasets_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        min_window_size: Minimum window length of loaded sequences.
        max_window_size: Maximum window length of loaded sequences.
        pad: If True, repeat last frame such that all sequences have length 'max_window_size'.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
        # TODO act_step actually is fwd_pred_next_n but not be rightly forward
    """

    def __init__(
        self,
        data_dir: Path,
        proprio_state: DictConfig = prop_state,
        lang_folder: str = "lang_annotations",
        num_workers: int = 0,
        key: str = "lang",
        obs_space: DictConfig = obs_config,
        transforms: Dict = {},
        batch_size: int = 32,
        window_size: int = 16,
        window_sample: Literal["sliding", "random", "range"]="sliding",
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        rgb_pad=-1,
        gripper_pad=-1,
        traj_cons=True,
        text_aug=False,
        dif_ws=False,
        act_step=1,
        norm_action=False,
        norm_min=-1,
        norm_max=1,
        regular_action=False,
        x_mean=0,
        x_std=1,
        **kwargs
    ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        self.window_sample = window_sample
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            self.min_window_size = min_window_size
            self.max_window_size = max_window_size
        self.act_step = act_step

        self.norm_action = norm_action
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.regular_action = regular_action
        self.x_mean = x_mean
        self.x_std = x_std

        # print('ws {}, min_ws {}, max_ws {}'.format(self.window_size, self.max_window_size, self.min_window_size))
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        # print(data_dir)
        self.abs_datasets_dir = data_dir
        if 'calvin_data_copy' in str(self.abs_datasets_dir):
            lang_folder = 'lang_annotations_test'
        self.lang_folder = lang_folder  # if self.with_lang else None
        self.aux_lang_loss_window = aux_lang_loss_window
        self.traj_cons = traj_cons

        self.text_aug = text_aug

        self.rgb_pad = rgb_pad
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)

        assert (
            "validation" in self.abs_datasets_dir.as_posix()
            or "training" in self.abs_datasets_dir.as_posix()
        )
        self.validation = "validation" in self.abs_datasets_dir.as_posix()
        assert self.abs_datasets_dir.is_dir()
        logger.info(f"loading dataset at {self.abs_datasets_dir}")
        logger.info("finished loading dataset")

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()
            
            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)
        return {"rgb_obs": seq_rgb_obs_dict}

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]}
    
    def discretize_action_bins(self, action, action_bin=256):
        
        action_min = -1.001
        action_max = 1.001
        action_len = (action_max - action_min) / action_bin
        action = torch.FloatTensor(action)
        pose_action = (pose_action - action_min) / action_len
        pose_action = torch.floor(pose_action).long().view(-1).tolist()
        pose_action[-1] = int(action[-1])
        return pose_action
    
    def process_rt2_ag_text(self, text, action):
        
        action_id = self.discretize_action_bins(action)
        action_text = ["<Action_{}>".format(i) for i in action_id]
        action_text.append("<Gripper_{}>".format(action[-1]))
        
        return action_text

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        head = False
        sequence = self._get_sequences(idx)

        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size, head=head)

        import copy
        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        
        image_tensors = self.image_fn(new_list)
        if self.rgb_pad != -1:
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(image_tensors.unsqueeze(0)).squeeze(0)
            else:
                image_tensors = self.rgb_shift(image_tensors)
        sequence["rgb_obs"]["rgb_static"] = image_tensors
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        
        gripper_tensors = self.image_fn(new_list)
        if self.gripper_pad != -1:
            if self.traj_cons:
                gripper_tensors = self.rgb_shift.forward_traj(gripper_tensors.unsqueeze(0)).squeeze(0)
            else:
                gripper_tensors = self.rgb_shift(gripper_tensors)
        sequence["rgb_obs"]["rgb_gripper"] = gripper_tensors
        # print(pad_size, len(new_list))
        return sequence

    def _get_sequences(self, idx: int) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx)

        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        info = self._add_language_info(info, idx)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  # type:ignore
        seq_dict["idx"] = idx  # type:ignore
        seq_dict['action_mask'] = episode['action_mask']
        return seq_dict

    def _load_episode(self, idx: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _get_window_size(self, idx: int) -> int:
        """
        Sample a window size taking into account the episode limits.

        Args:
            idx: Index of the sequence to load.

        Returns:
            Window size.
        """
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            # last episode
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif (
            self.episode_lookup[idx + window_diff]
            != self.episode_lookup[idx] + window_diff
        ):
            # less than max_episode steps until next episode
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx : idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(
                self.max_window_size, (self.min_window_size + steps_to_next_episode - 1)
            )
        else:
            max_window = self.max_window_size

        if self.validation:
            # in validation step, repeat the window sizes for each epoch.
            return get_validation_window_size(idx, self.min_window_size, max_window)
        else:
            return np.random.randint(self.min_window_size, max_window + 1)

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                        self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )
        return seq

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info


class DiskCalvinVideoDataset(BaseCalvinDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        image_fn: Callable,
        tokenizer: Callable,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        partial_data=False,
        decoder_type='lstm',
        discrete_action=False,
        discrete_action_history=False,
        action_tokenizer=None,
        model_name='vicuna',
        predict_stop_token=True,
        use_mu_law=False,
        n_bin=256,
        min_action=-1,
        max_action=1,
        clip_tokenizer=None,
        task_type='calvin_action',
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.decoder_type = decoder_type
        self.save_format = save_format
        self.image_fn = image_fn
        if isinstance(tokenizer, dict):
            tokenizer_type = tokenizer['tokenizer_type']
            max_text_len = tokenizer['max_text_len']
            tokenizer = build_tokenizer(tokenizer_config=tokenizer)
            self.tokenizer = tokenizer
            self.text_fn = get_text_function(tokenizer, tokenizer_type, max_text_len)
        else:
            self.text_fn = tokenizer
        self.partial_data = partial_data
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.clip_tokenizer = clip_tokenizer
        self.pretrain = pretrain
        self.skip_frames = skip_frames
        self.use_mu_law = use_mu_law
        self.task_type= task_type
        
        (
            self.episode_lookup,
            self.left_pad_lookup,
            self.right_pad_lookup,
            self.lang_lookup,
            self.lang_ann,
            self.lang_task
        ) = self._build_file_indices_lang(self.abs_datasets_dir)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(
            self.abs_datasets_dir, self.save_format
        )

        self.model_name = model_name
        self.discrete_action = discrete_action
        self.discrete_action_history = discrete_action_history
        self.predict_stop_token = predict_stop_token
        if self.discrete_action:
            if action_tokenizer is None:
                action_tokenizer = ActionTokenizer(self.tokenizer, bins=n_bin, min_action=min_action, max_action=max_action)
            self.action_tokenizer = action_tokenizer

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.
        Args:
            file_idx: index of starting frame.
        Returns:
            Path to file.
        """
        return Path(
            f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def _load_episode(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # start_idx = self.episode_lookup[idx]
        # end_idx = start_idx + window_size
        end_idx = self.episode_lookup[idx]
        start_idx = end_idx - self.window_size - self.act_step + 2
        left_pad_len = self.left_pad_lookup[idx]
        right_pad_len = self.right_pad_lookup[idx]
        idx_range = np.array(range(start_idx, end_idx+1))
        action_mask = np.ones_like(idx_range)
        if left_pad_len > 0:
            idx_range[:left_pad_len] = idx_range[left_pad_len]
            action_mask[:left_pad_len] = 0
        if right_pad_len < 0:
            idx_range[right_pad_len:] = idx_range[right_pad_len-1]
            action_mask[right_pad_len:] = 0

        # end_idx = self.episode_lookup[idx]
        # start_idx = end_idx - window_size
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episodes = [
            self.load_file(self._get_episode_name(file_idx))
            for file_idx in idx_range
        ]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}

        episode["language"] = self.lang_ann[self.lang_lookup[idx]]
        if left_pad_len > 0:
            episode["rel_actions"][:left_pad_len] = 0
        if right_pad_len < 0:
            episode['rel_actions'][right_pad_len:] = 0
        episode['action_mask'] = action_mask.astype(bool)
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []
        left_pad_lookup = []
        right_pad_lookup = []

        try:
            print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                allow_pickle=True,
            ).item()
        except Exception:
            print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
            ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]
        lang_lookup = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            start_idx = start_idx + self.act_step - 1
            end_idx = end_idx + self.act_step - 1
            assert end_idx >= self.max_window_size
            if end_idx <= start_idx:
                continue
            cnt = 0
            left_pad = self.window_size - 1
            right_pad = end_idx - start_idx - self.act_step
            
            for idx in range(start_idx, end_idx):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                    right_pad_lookup.append(min(right_pad, 0))

                    if self.window_sample == "random":
                        left_pad_lookup.append(left_pad % self.window_size)
                    else:
                        left_pad = max(0, left_pad)
                        left_pad_lookup.append(left_pad)
                        if self.window_sample == "range":
                            for tmp_pad_len in range(left_pad+1, self.window_size):
                                episode_lookup.append(idx)
                                lang_lookup.append(i)
                                right_pad_lookup.append(min(right_pad, 0))
                                left_pad_lookup.append(tmp_pad_len)
                        
                left_pad -= 1
                right_pad -= 1
                cnt += 1

        return np.array(episode_lookup), np.array(left_pad_lookup), np.array(right_pad_lookup), lang_lookup, lang_ann, lang_task

    # def wrap_instruction_and_action(self, lang, action, action_mask):
    #     # modified from OpenVLA
    #     IGNORE_INDEX = -100
    #     prompt_builder = get_prompt_builder(self.model_name, eos=self.tokenizer.eos_token, bos=self.tokenizer.bos_token)
    #     # if pass in multi-step actions, we concat them
    #     assert action.shape[0] == self.act_step + self.window_size - 1
        
    #     if self.discrete_action_history:
    #         history_action = action[: self.window_size - 1]
    #         history_mask = action_mask[: self.window_size - 1]
    #         history_action = history_action[history_mask]
    #         history_len = history_mask.sum()
    #     else:
    #         history_action = np.zeros((0, action.shape[1]))
    #         history_len = 0
        
    #     next_action = action[-self.act_step:]
    #     next_action_mask = action_mask[-self.act_step:]
    #     next_len = next_action_mask.sum()
        
    #     action_dim = action.shape[1]
    #     history_action = history_action.flatten()
    #     next_action = next_action.flatten()
        
    #     if history_len == 1:
    #         history_prompt = f"Here is {history_len} step action that the robot has taken: "
    #     elif history_len > 1:
    #         history_prompt = f"Here are {history_len} step actions that the robot has taken: "
    #     else:
    #         history_prompt = ""
        
    #     if self.act_step == 1:
    #         question_prompt = f"What action should the robot take to {lang}?"
    #     else:
    #         question_prompt = f"What {self.act_step} step actions should the robot take to {lang}?"

    #     conversation = [
    #         {"from": "human", "value": history_prompt + question_prompt},
    #         {"from": "gpt", "value": ""},
    #     ]
        
    #     input_ids = []
    #     for turn in conversation:
    #         prompt_builder.add_turn(turn["from"], turn["value"])
    #     prompt: str = prompt_builder.get_prompt()

    #     add_special_tokens = True
    #     if history_len > 0:
    #         prefix_index = prompt.find(history_prompt)+len(history_prompt)
    #         prefix_prompt = prompt[:prefix_index]
    #         prompt = prompt[prefix_index:]
    #         input_ids = self.tokenizer(prefix_prompt, add_special_tokens=add_special_tokens).input_ids
    #         add_special_tokens = False
    #         history_action_ids = self.action_tokenizer.encode_actions_to_token_ids(history_action)
    #         input_ids += history_action_ids
            
    #     input_ids += self.tokenizer(prompt, add_special_tokens=add_special_tokens).input_ids
    #     next_action_ids = self.action_tokenizer.encode_actions_to_token_ids(next_action)
    #     if self.tokenizer.eos_token is None:
    #         input_ids = input_ids + next_action_ids
    #     else:
    #         input_ids = input_ids[:-1] + next_action_ids + [input_ids[-1]]
        
    #     labels = list(input_ids)
    #     input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

    #     # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
    #     right_pad_len = (self.act_step - next_len) * action_dim
    #     if self.tokenizer.eos_token is None:
    #         labels[: -self.act_step*action_dim] = IGNORE_INDEX
    #         if right_pad_len != 0:
    #             labels[-right_pad_len:] = IGNORE_INDEX
    #     else:
    #         labels[: -(self.act_step*action_dim + 1)] = IGNORE_INDEX
    #         if right_pad_len != 0:
    #             labels[-right_pad_len-1: -1] = IGNORE_INDEX
        
    #     if not self.predict_stop_token and self.tokenizer.eos_token:
    #         labels[-1] = IGNORE_INDEX

    #     return input_ids, labels

    @staticmethod
    def static_wrap_instruction_and_action(
        lang, action, action_mask, 
        model_name, tokenizer, action_tokenizer,
        act_step, window_size, discrete_action_history, predict_stop_token,
        mode="train"
    ):
        # modified from OpenVLA
        IGNORE_INDEX = -100
        prompt_builder = get_prompt_builder(model_name, eos=tokenizer.eos_token, bos=tokenizer.bos_token)
        # if pass in multi-step actions, we concat them
        if mode == "train":
            assert action.shape[0] == act_step + window_size - 1
        else:
            assert action.shape[0] == window_size - 1
        if discrete_action_history:
            history_action = action[: window_size - 1]
            history_mask = action_mask[: window_size - 1]
            history_action = history_action[history_mask]
            history_len = history_mask.sum()
        else:
            history_action = np.zeros((0, action.shape[1]))
            history_len = 0
        
        next_action = action[-act_step:]
        next_action_mask = action_mask[-act_step:]
        next_len = next_action_mask.sum()
        
        action_dim = action.shape[1]
        history_action = history_action.flatten()
        next_action = next_action.flatten()
        
        if history_len == 1:
            history_prompt = f"Here is {history_len} step action that the robot has taken: "
        elif history_len > 1:
            history_prompt = f"Here are {history_len} step actions that the robot has taken: "
        else:
            history_prompt = ""
        
        if act_step == 1:
            question_prompt = f"What action should the robot take to {lang}?"
        else:
            question_prompt = f"What {act_step} step actions should the robot take to {lang}?"

        conversation = [
            {"from": "human", "value": history_prompt + question_prompt},
            {"from": "gpt", "value": ""},
        ]
        
        input_ids = []
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        prompt: str = prompt_builder.get_prompt()

        add_special_tokens = True
        if history_len > 0:
            prefix_index = prompt.find(history_prompt)+len(history_prompt)
            prefix_prompt = prompt[:prefix_index]
            prompt = prompt[prefix_index:]
            input_ids = tokenizer(prefix_prompt, add_special_tokens=add_special_tokens).input_ids
            add_special_tokens = False
            history_action_ids = action_tokenizer.encode_actions_to_token_ids(history_action)
            input_ids += history_action_ids
        
        input_ids += tokenizer(prompt, add_special_tokens=add_special_tokens).input_ids
        if tokenizer.eos_token is not None and tokenizer.eos_token_id != input_ids[-1]:
            input_ids = input_ids + [tokenizer.eos_token_id]
        if mode == "train":
            next_action_ids = action_tokenizer.encode_actions_to_token_ids(next_action)
            if tokenizer.eos_token is None:
                input_ids = input_ids + next_action_ids
            else:
                input_ids = input_ids[:-1] + next_action_ids + [input_ids[-1]]
        elif tokenizer.eos_token is not None:
            input_ids = input_ids[:-1]
                
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        if mode == "train":
            right_pad_len = (act_step - next_len) * action_dim
            if tokenizer.eos_token is None:
                labels[: -act_step*action_dim] = IGNORE_INDEX
                if right_pad_len != 0:
                    labels[-right_pad_len:] = IGNORE_INDEX
            else:
                labels[: -(act_step*action_dim + 1)] = IGNORE_INDEX
                if right_pad_len != 0:
                    labels[-right_pad_len-1: -1] = IGNORE_INDEX
            
            if (not predict_stop_token and tokenizer.eos_token) or right_pad_len != 0:
                labels[-1] = IGNORE_INDEX

        return input_ids, labels

    def collater(self, sample):
        # forward transformation of mu law companding
        
        if self.norm_action:
            new_sample = []
            for s in sample:
                s['actions'] = normalize_action(s['actions'], self.norm_min, self.norm_max, maintain_last=True)
                new_sample.append(s)
            sample = new_sample

        if self.regular_action:
            new_sample = []
            for s in sample:
                s['actions'] = regularize_action(s['actions'], self.x_mean, self.x_std)
                new_sample.append(s)
            sample = new_sample

        if self.use_mu_law:
            new_sample = []
            for s in sample:
                s['actions'] = mu_law_companding(s['actions'])
                new_sample.append(s)
            sample = new_sample
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        action_tensors[..., -1] = ((action_tensors[..., -1] + 1) // 2).float()
        action_masks = torch.from_numpy(np.array([s["action_mask"] for s in sample]))
        action_chunck = action_tensors[:, -self.act_step:]
        action_chunck_mask = action_masks[:, -self.act_step:]

        history_action = action_tensors[:, :self.window_size-1]
        
        # raw_images = torch.stack([torch.from_numpy(np.stack(s["rgb_obs"]["rgb_static"])) for s in sample])
        # image_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        # gripper_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        
        raw_images = None
        image_tensors = torch.stack([s["rgb_obs"]["rgb_static"] for s in sample])
        gripper_tensors = torch.stack([s["rgb_obs"]["rgb_gripper"] for s in sample])

        # if self.act_step != 1:
        #     image_tensors = image_tensors[:, :-(self.act_step-1)]
        #     gripper_tensors = gripper_tensors[:, :-(self.act_step-1)]
        # if self.rgb_pad != -1:
        #     bs, seq_len = image_tensors.shape[:2]
        #     if self.traj_cons:
        #         image_tensors = self.rgb_shift.forward_traj(image_tensors)
        #     else:
        #         image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
        #         image_tensors = self.rgb_shift(image_tensors)
        #         image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
        # if self.gripper_pad != -1:
        #     bs, seq_len = gripper_tensors.shape[:2]
        #     if self.traj_cons:
        #         gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors)
        #     else:
        #         gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
        #         gripper_tensors = self.gripper_shift(gripper_tensors)
        #         gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])

        stacked_language = [s["lang"] for s in sample]
        text_tensors, text_attention_mask = self.text_fn(stacked_language)
        
        clip_text = None
        if self.clip_tokenizer is not None:
            clip_text = self.clip_tokenizer(stacked_language)
        
        bs = len(sample)
        instr_and_action_ids = None
        instr_and_action_labels = None
        instr_and_action_mask = None
        
        if self.discrete_action:
            res = [
                self.static_wrap_instruction_and_action(
                    s["lang"], s["actions"], s["action_mask"],
                    self.model_name, self.tokenizer, self.action_tokenizer,
                    self.act_step, self.window_size, self.discrete_action_history,
                    self.predict_stop_token, mode="train"
                ) for s in sample
            ]
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
        res = {
            "rgb": image_tensors,
            "hand_rgb": gripper_tensors,
            "text": text_tensors,
            "text_mask": text_attention_mask,
            "clip_text": clip_text,
            "fwd_rgb_chunck": image_tensors.clone().unsqueeze(2),
            "fwd_hand_rgb_chunck": gripper_tensors.clone().unsqueeze(2),
            "action_chunck": action_chunck,
            "chunck_mask": action_chunck_mask,
            "raw_image": raw_images,
            "instr_and_action_ids": instr_and_action_ids,
            "instr_and_action_labels": instr_and_action_labels,
            "instr_and_action_mask": instr_and_action_mask,
            "raw_text": stacked_language,
            "history_action": history_action,
            "data_source": self.task_type
        }
        return res
    

def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None
    dataset: Dataset = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def preprocess_image(sample, image_processor):
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip and color jitter
    return image


def preprocess_text_calvin(sample, tokenizer, decoder_type='lstm'):
    tokenizer.padding_side = "right"
    max_length = 48 if decoder_type == 'rt2_enc' else 32
    if decoder_type == 'rt2_enc':
        action_str = ''.join([f'<Action_{i}>' for i in range(7)])
        sample = [
            (f"<image>{s.strip()}{action_str}<|endofchunk|>{tokenizer.eos_token}") for s in sample
        ]
    
    else:
        sample = [
            (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample
        ]
    text = tokenizer(
        sample,
        max_length=max_length,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )
    return text["input_ids"], text["attention_mask"]


def preprocess_interleaved(sample, tokenizer, clip_processor, sim_threshold):
    info = json.loads(sample[0])
    tar_file_obj = io.BytesIO(sample[1])
    image_tar = tarfile.open(fileobj=tar_file_obj)
    sentences = info["text_list"]

    images, image_idxs = [], []
    for image_path, sim in zip(info["image_info"], info["similarity_matrix"]):
        # pick one image per sentence
        if info["image_info"][image_path]["matched_text_index"] in image_idxs:
            continue
        rawbytes = image_tar.extractfile(
            os.path.join(image_tar.getnames()[0], image_path)
        ).read()

        # filter to images >= 10KB
        if len(rawbytes) // 1000 <= MIN_KB:
            continue
        if sim[info["image_info"][image_path]["matched_text_index"]] < sim_threshold:
            continue
        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")

        images.append(image)
        image_idxs.append(info["image_info"][image_path]["matched_text_index"])

    if len(images) == 0:
        raise ValueError("No images in sample")

    # filter out images that are exact duplicates
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), MAX_NUM_IMAGES))
    images_tensors = images_tensors[keep_ixs]
    image_idxs = [image_idxs[ix] for ix in keep_ixs]

    # pad to 5 images
    if len(images_tensors) < MAX_NUM_IMAGES:
        zero_padding = torch.zeros(
            (MAX_NUM_IMAGES - len(images_tensors), 3, 224, 224), dtype=torch.float
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # add in <image> and <eoc> tokens
    # eoc after sentence = "sentence loss"
    for ix in image_idxs:
        sentences[ix] = f"<|endofchunk|><image>{sentences[ix]}"

    text = " ".join(sentences)
    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )
    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text, max_length=256, truncation=True, padding="max_length", return_tensors="pt"
    )

    # reject sequences with too few images (after truncation)
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )

    if num_images == 0:
        raise ValueError("No images in sample")
    elif (
        num_images == 1 and random.random() <= 0.5
    ):  # 50% chance of keeping single image samples
        raise ValueError("Only one image in sample")

    return (
        images_tensors,
        (text_tensor["input_ids"], text_tensor["attention_mask"]),
    )

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import clip
    from pathlib import Path
    data_path = "/mnt/bn/robotics-real-data/data_gr2_zhb/anns/ego4d/gr2-1121/train/json/"
    data_path = "/mnt/bn/robotics-data-hl/real_data/gr2_labels_1110/CALVIN/task_ABCD_D/val"
    # data_path = "/mnt/bn/robotics-lq2024/zhb/gr2_data/anns/ego4d/gr2-1130/train/json"
    # data_path = "/mnt/bn/robotics-lq2024/zhb/gr2_data/anns/ego4d/gr2-1120compressed/train/json"
    # _, clip_preprocess = clip.load('ViT-B/32', device='cpu')
    from torchvision import transforms
    # clip_preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # config = {
    #     "type": "DiskCalvinDataset",
    #     "data_dir": "/mnt/bn/robotics/manipulation_data/calvin_data/task_ABCD_D/validation",
    #     "shift_first": False,
    #     "window_size": 5,
    #     "tokenizer": {
    #         "type": "AutoTokenizer",
    #         "pretrained_model_name_or_path": "/mnt/bn/robotics-data-lxh-lq/LLaVA/llava-v1.6-vicuna-7b",
    #         "tokenizer_type": "flamingo",
    #         "max_text_len": 128,
    #         # "additional_special_tokens": ["<|endofchunk|>", "<image>"]
    #     }
    # }
    # import functools
    # def preprocess_image(sample, image_processor):
    #     image = [image_processor(s).unsqueeze(0) for s in sample]
    #     image = torch.cat(image, dim=0)
    #     # apply random horizontal flip and color jitter
    #     return image
    # import pdb; pdb.set_trace()
    # dataset = DiskCalvinVideoDataset(
    #     data_dir=config["data_dir"],
    #     tokenizer=config["tokenizer"],
    #     window_size=config["window_size"],
    #     image_fn=functools.partial(preprocess_image, image_processor=clip_preprocess),
    #     action_space='discrete',
    #     discrete_action=True,
    #     discrete_action_history=False,
    #     model_name='vicuna',
    #     act_step=3,
    #     n_bins=512
    # )
    # config_path = "training_dataset_config.pkl"
    config_path = "DiskCalvinVideoDataset.pkl"
    with open(config_path, 'rb') as file:
        import pickle as pkl
        config = pkl.load(file)
    import pdb; pdb.set_trace()
    dataset = DiskCalvinVideoDataset(**config)
    datas = [dataset[0], dataset[1], dataset[62], dataset[63]]
    dataset.collater(datas)

    # dataset = DiskCalvinVideoDataset(
    #     data_dir=config["data_dir"],
    #     tokenizer=config["tokenizer"],
    #     window_size=config["window_size"],
    #     image_fn=functools.partial(preprocess_image, image_processor=clip_preprocess),
    #     action_space='discrete',
    #     discrete_action=True,
    #     discrete_action_history=True,
    #     model_name='vicuna',
    #     window_sample="sliding",
    #     act_step=3,
    #     n_bins=512
    # )
    # datas = [dataset[0], dataset[1], dataset[62], dataset[63]]
    # dataset.collater(datas)
    
    # dataset = DiskCalvinVideoDataset(
    #     data_dir=config["data_dir"],
    #     tokenizer=config["tokenizer"],
    #     window_size=config["window_size"],
    #     image_fn=functools.partial(preprocess_image, image_processor=clip_preprocess),
    #     action_space='discrete',
    #     discrete_action=True,
    #     discrete_action_history=True,
    #     model_name='vicuna',
    #     window_sample="random",
    #     act_step=3,
    #     n_bins=512
    # )
    # datas = [dataset[0], dataset[1], dataset[62], dataset[63]]
    # dataset.collater(datas)


    dataset = DiskCalvinVideoDataset(
        data_dir=config["data_dir"],
        tokenizer=config["tokenizer"],
        window_size=config["window_size"],
        image_fn=functools.partial(preprocess_image, image_processor=clip_preprocess),
        action_space='discrete',
        discrete_action=True,
        discrete_action_history=True,
        model_name='vicuna',
        window_sample="range",
        act_step=3,
        n_bins=512
    )
    datas = [dataset[0], dataset[1], dataset[62], dataset[63]]
    dataset.collater(datas)

    exit()

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=dataset.collater)
    for i, data in enumerate(dataloader):
        for d in data:
            if hasattr(data[d], 'shape'):
                print(data[d].shape)
            # print(d, data[d])
            continue
        # exit(0)
        import pdb; pdb.set_trace()
    print('start loading clip')
    _, clip_preprocess = clip.load('ViT-B/32', device='cpu')
    print('clip loaded')
    from transformers import AutoTokenizer
    import functools
    
    def preprocess_text_calvin(sample, tokenizer):
        tokenizer.padding_side = "right"
        sample = [
            # (f"{s.strip()}{tokenizer.eos_token}")
            # for s in sample
            (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample
        ]
        text = tokenizer(
            sample,
            max_length=32,
            padding="longest",
            truncation="only_first",
            return_tensors="pt",
        )
        return text["input_ids"], text["attention_mask"]

    tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/robotics/lxh/mpt-1b-redpajama-200b-dolly", local_files_only=True)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    seq_len = 64
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    dataset = GRDataset(
        data_dir=data_path,
        # text_fn=preprocess_text_fn,
        tokenizer=tokenizer,
        preprocess=None,
        seq_len=seq_len,
        fwd_pred_next_n=1,
        mode='train',
        obs_mode='image'
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=dataset.collater)
    for i, data in enumerate(dataloader):
        for d in data:
            print(d, data[d])
            continue
            if d is None:
                continue
            if isinstance(d, tuple):
                print(d[0].shape, d[1].shape)
            else:
                print(d.shape)
            print(d)
            # print(d, data[d])
        # exit(0)
    pass