import pickle
import numpy as np
from omegaconf import DictConfig
from typing import Dict
from pathlib import Path
from abc import ABC, abstractmethod
from calvin_agent.datasets.utils.episode_utils import lookup_naming_pattern

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

def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())

class BaseCalvinDataset(ABC):
    def __init__(
        self,
        data_dir: Path,
        proprio_state: DictConfig = prop_state,
        lang_folder: str = "lang_annotations",
        obs_space: DictConfig = obs_config,
        save_format: str = "npz",
        **kwargs
    ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.abs_datasets_dir = data_dir
        self.lang_folder = lang_folder  # if self.with_lang else None
        assert (
            "validation" in self.abs_datasets_dir.as_posix()
            or "training" in self.abs_datasets_dir.as_posix()
        )
        self.validation = "validation" in self.abs_datasets_dir.as_posix()
        assert self.abs_datasets_dir.is_dir()
        self.save_format = save_format

        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError

        self.naming_pattern, self.n_digits = lookup_naming_pattern(
            self.abs_datasets_dir, self.save_format
        )
        self._build_file_indices_lang(self.abs_datasets_dir)

    def _get_episode_name(self, file_idx: int) -> Path:
        return Path(
            f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )
    
    @abstractmethod
    def _build_file_indices_lang(self, abs_datasets_dir: Path):
        raise NotImplementedError("You must implement function _build_file_indices_lang ")

    @abstractmethod
    def _load_episode(self, idx: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError("You must implement function _load_episode")


class CalvinGlobalDataset(BaseCalvinDataset):
    def __init__(
        self,
        frame_num: int,
        **kwargs
    ):
        self.frame_num = frame_num
        super().__init__(**kwargs)

    def _build_file_indices_lang(self, abs_datasets_dir: Path):
        assert abs_datasets_dir.is_dir()
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

        ep_start_end_ids = np.array(lang_data["info"]["indx"])  # each of them are 64
        lang_ann = np.array(lang_data["language"]["ann"])  # length total number of annotations

        # the episode length must be larger than frame_num
        episode_len = ep_start_end_ids[..., 1] - ep_start_end_ids[..., 0]
        episode_mask = (episode_len >= self.frame_num)
        ep_start_end_ids = ep_start_end_ids[episode_mask]
        lang_ann = lang_ann[episode_mask]

        start_ids = ep_start_end_ids[..., 0]
        end_ids = ep_start_end_ids[:, 1]
        
        t = np.linspace(0, 1, self.frame_num)
        episode_frame_index_array = np.expand_dims(start_ids, axis=-1) * (1 - t) + np.expand_dims(end_ids, axis=-1) * t
        episode_frame_index_array = episode_frame_index_array.astype(int)

        self.frame_idx_array, self.lang_ann = episode_frame_index_array, lang_ann

    def _load_episode(self, idx: int) -> Dict[str, np.ndarray]:
        file_idx_list = self.frame_idx_array[idx]
        lang_ann = self.lang_ann[idx]

        episodes = [
            self.load_file(self._get_episode_name(file_idx))
            for file_idx in file_idx_list
        ]
        static_images = np.stack([ep["rgb_static"] for ep in episodes])
        gripper_images = np.stack([ep["rgb_gripper"] for ep in episodes])
        
        return dict(
            static_images=static_images, 
            gripper_images=gripper_images, 
            lang_ann=lang_ann,
        )

