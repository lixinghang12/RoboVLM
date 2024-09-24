import logging
import threading
import queue
import numpy as np
from PIL import Image
from omegaconf import DictConfig
import pyhash
import tqdm

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
from typing import Dict, Tuple, Union

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


from typing import Any, Dict, List, Tuple, Callable
from itertools import chain
from calvin_agent.datasets.utils.episode_utils import lookup_naming_pattern
import pickle
import torch.nn as nn
import torch.nn.functional as F


class BaseCalvinDataset:
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
    """

    def __init__(
        self,
        data_dir: Path,
        proprio_state: DictConfig = prop_state,
        num_workers: int = 0,
        obs_space: DictConfig = obs_config,
        gripper_pad=-1,
        act_step=1,
    ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.relative_actions = "rel_actions" in self.observation_space["actions"]

        self.num_workers = num_workers
        self.act_step = act_step
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.abs_datasets_dir = data_dir
        self.gripper_pad = gripper_pad

        assert (
            "validation" in self.abs_datasets_dir.as_posix()
            or "training" in self.abs_datasets_dir.as_posix()
        )
        assert self.abs_datasets_dir.is_dir()
        self.save_format = 'npz'
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)
        self.naming_pattern, self.n_digits = lookup_naming_pattern(
            self.abs_datasets_dir, self.save_format
        )
        
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

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")

        for start_idx, end_idx in ep_start_end_ids:
            episode_lookup.append(np.arange(start_idx, end_idx + 1))
        episode_lookup = np.concatenate(episode_lookup, axis=0)
        return episode_lookup

    def get_action_slidce(self, idx_list, result_queue, pbar):
        for idx in tqdm.tqdm(idx_list):
            result_queue.put(self._load_action(idx))
            pbar.update(1)

    def get_actions(self):
        action_queue = queue.Queue()
        # for idx in tqdm.tqdm(self.episode_lookup):
        #     action = self._load_action(idx)
        #     action_list.append(action)
        threads = []
        num_threads = 10
        idxs = list(range(len(self.episode_lookup)))
        pbar = tqdm.tqdm(len(self.episode_lookup))
        for i in range(num_threads):
            idx_list = idxs[i::num_threads]
            t = threading.Thread(target=self.get_action_slidce, args=(idx_list, action_queue, pbar))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        actions = []
        while not action_queue.empty():
            actions.append(action_queue.get())

        actions = np.stack(actions, axis=0)
        return actions
        
    def count(self):
        file_path = self.abs_datasets_dir / "ep_start_end_ids.npy"
        if not file_path.exists():
            actions = self.get_actions()
            np.save(file_path, actions)
        else:
            actions = np.load(self.abs_datasets_dir / 'actions.npy')
        
        xyz_actions = actions[..., :3]
        rpy_actions = actions[..., 3:6]
        xyz_actions /= 50
        rpy_actions /= 20
        self.count_statics(xyz_actions, rpy_actions)
        
    @staticmethod
    def count_statics(xyz_actions, rpy_actions):
        from IPython import embed; embed()
        print("Mean of xyz actions:")
        print(np.abs(xyz_actions).mean())
        print("Mean of rpy actions:")
        print(np.abs(rpy_actions).mean())
        max_pos = 0.02
        max_orn = 0.05
        print("Over the max xyz action bound: ")
        print(np.mean(np.abs(xyz_actions) > max_pos))
        print("Over the max rpy action bound: ")
        print(np.mean(np.abs(rpy_actions) > max_orn))
        print("xyz action mean in the bound:")
        print(np.mean(np.abs(np.clip(xyz_actions, -max_pos, max_pos))))
        print("rpy action mean in the bound:")
        print(np.mean(np.abs(np.clip(rpy_actions, -max_orn, max_orn))))

    def _load_action(self, idx: int) -> Dict[str, np.ndarray]:
        file_idx = self.episode_lookup[idx]
        episode = self.load_file(self._get_episode_name(file_idx))
        action = episode['rel_actions']
        return action

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


if __name__ == "__main__":
    config = {
        "type": "DiskCalvinDataset",
        "data_dir": "/mnt/bn/robotics/manipulation_data/calvin_data/task_ABCD_D/training",
        "shift_first": False,
        "window_size": 5,
        "tokenizer": {
            "type": "AutoTokenizer",
            "pretrained_model_name_or_path": "/mnt/bn/robotics/lxh/mpt-1b-redpajama-200b-dolly",
            "tokenizer_type": "flamingo",
            "max_text_len": 32,
            "additional_special_tokens": ["<|endofchunk|>", "<image>"]
        }
    }
    
    dataset = BaseCalvinDataset(
        data_dir=config["data_dir"],
    )
    dataset.count()