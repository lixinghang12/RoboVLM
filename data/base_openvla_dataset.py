"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""
from functools import partial
from pathlib import Path
from typing import Dict, Literal, Any
import torch
from torch.utils.data import IterableDataset

class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        image_size: int,
        chunk_action: bool=True,
        frame_num: int=-1,
        left_pad: bool=False,
        window_sample: Literal['sliding', 'range']='sliding',
        window_size: int=1,
        fwd_pred_next_n: int=1,
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        **kwargs
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
        from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
        super().__init__()
        self.data_root_dir, self.data_mix = data_root_dir, data_mix
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                                      # If we wanted to feed / predict more than one step
                chunk_action=chunk_action,
                frame_num=frame_num,
                future_action_window_size=fwd_pred_next_n,                        # For action chunking
                left_pad=left_pad,
                window_sample=window_sample,
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=(image_size, image_size),
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        from prismatic.vla.datasets.rlds import make_interleaved_dataset
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield rlds_batch

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


if __name__ == "__main__":
    import clip
    _, clip_preprocess = clip.load('ViT-B/32', device='cpu')
    image_size = 224
    def preprocess_image(sample, image_processor):
        image = [image_processor(s).unsqueeze(0) for s in sample]
        image = torch.cat(image, dim=0)
        # apply random horizontal flip and color jitter
        return image

    config = dict(
        data_root_dir="/mnt/bn/robotics-data-lxh-lq/openvla/datasets/open-x-embodiment",
        # data_mix="bridge",
        data_mix="rt_1",
        image_size=image_size,
        window_size=8,
        fwd_pred_next_n=2,
        image_fn=partial(preprocess_image, image_processor=clip_preprocess),
        model_name="qwen",
        tokenizer={
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
        },
        predict_stop_token=True,
        traj_cons=True,
        shuffle_buffer_size=256_000,
        train=True,
        rgb_pad=2,
        gripper_pad=2,
        discrete=True,
        discrete_action_history=True,
        image_aug=True,
        norm_action=False,
        norm_min=-1,
        norm_max=1,
        regular_action=False,
        x_mean=0,
        x_std=1,
        n_bin=256,
        min_action=-1,
        max_action=1,
    )

    for discrete in [True, False]:
        for organize_type in ["interleave", "segment"]:
            if organize_type == "interleave":
                left_pad = False
                window_sample = "sliding"
            else:
                left_pad = True
                window_sample = "range"
            
            if discrete:
                action_history_list = [True, False]
            else:
                action_history_list = [True]

            for discrete_action_history in action_history_list:
                config['discrete'] = discrete
                config['left_pad'] = left_pad
                config['window_sample'] = window_sample
                config['discrete_action_history'] = discrete_action_history
                print(
                    f"discrete: {discrete}\n"
                    f"left_pad: {left_pad}\n"
                    f"window_sample: {window_sample}\n"
                    f"discrete_action_history: {discrete_action_history}\n"
                )
                dataset = RLDSDataset(**config)
                samples = []
                count = 0
                for sample in dataset.__iter__():
                    if count == 4:
                        break
                    samples.append(sample)
                    count += 1

                dataset.collater(samples)

                print("pass code test!!!\n")