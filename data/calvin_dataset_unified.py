from asyncore import file_dispatcher
import numpy as np
from itertools import chain
from typing import Dict, Literal
import pickle as pkl
from pathlib import Path
from data.base_action_prediction_dataset import ActionPredictionDataset
from data.base_calvin_dataset import BaseCalvinDataset
from data.data_utils import get_chunked_episode


class CalvinDataset(ActionPredictionDataset, BaseCalvinDataset):
    def __init__(
        self,
        skip_frames: int=1,
        **kwargs
    ):
        ActionPredictionDataset.__init__(self, **kwargs)
        if self.organize_type == "interleave":
            self.window_sample, self.left_pad = 'sliding', False
        elif self.organize_type == "segment":
            self.window_sample, self.left_pad = "range", True
        else:
            raise ValueError("organize type must be interleave or segment")
        self.skip_frames = skip_frames
        BaseCalvinDataset.__init__(self, **kwargs)

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

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        episode_chunk_lookup, chunk_mask_lookup, lang_lookup = [], [], []

        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            episode_idx_range = np.arange(start_idx, end_idx)[::self.skip_frames]
            chunked_episode_idx, chunk_mask = get_chunked_episode(self.window_sample, self.left_pad, self.window_size, self.fwd_pred_next_n, episode_idx_range)
            
            episode_chunk_lookup.append(chunked_episode_idx)
            chunk_mask_lookup.append(chunk_mask)
            lang_lookup.append(np.full(chunked_episode_idx.shape[0], fill_value=i))

        (
            self.episode_lookup,
            self.chunk_mask_lookup,
            self.lang_lookup,
            self.lang_ann,
        ) = (
            np.concatenate(episode_chunk_lookup, axis=0), 
            np.concatenate(chunk_mask_lookup, axis=0), 
            np.concatenate(lang_lookup, axis=0), 
            lang_ann,
        )

    def _load_episode(self, idx: int) -> Dict[str, np.ndarray]:
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.remove('robot_obs')

        file_idx_list = self.episode_lookup[idx]
        chunk_mask = self.chunk_mask_lookup[idx]
        episodes = [
            self.load_file(self._get_episode_name(file_idx))
            for file_idx in file_idx_list
        ]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}

        episode["language"] = self.lang_ann[self.lang_lookup[idx]]
        episode['action_mask'] = chunk_mask.astype(bool)
        return episode

    def __getitem__(self, idx: int) -> Dict:
        episode = self._load_episode(idx)
        return self.batch_transform(
            task_description=episode['language'],
            action=episode['rel_actions'],
            episode_mask=episode['action_mask'],
            images=episode['rgb_static'],
            gripper_images=episode['rgb_gripper'],
        )

    def __len__(self) -> int:
        return len(self.episode_lookup)


if __name__ == "__main__":
    config_path = "training_dataset_config.pkl"
    with open(config_path, 'rb') as file:
        config = pkl.load(file)
    # from IPython import embed; embed()
    dataset = CalvinDataset(**config)
    
    datas = [dataset[52], dataset[53], dataset[54], dataset[0]]
    dataset.collater(datas)
