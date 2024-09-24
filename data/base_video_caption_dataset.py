from dataclasses import dataclass
import numpy as np
from typing import Any, Dict, Callable, Union, Optional, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import PreTrainedTokenizerBase
from data.base_task_dataset import BaseTaskDataset, IGNORE_INDEX
from data.data_utils import get_prompt_builder, pad_sequences

@dataclass
class VideoCaptionBatchTransform:
    """
    Transform one item of dataset
    """
    model_name: str
    tokenizer: PreTrainedTokenizerBase
    text_fn: Callable
    image_fn: Callable
    
    predict_stop_token: bool

    def __call__(self, video: np.ndarray, gripper_video: Optional[np.ndarray], task_description: str) -> Dict[str, Any]:
        """Converts an item to the format expected by collator/models."""

        video_tensor = self.image_fn(video)
        gripper_video_tensor = self.image_fn(gripper_video, static=False) if gripper_video is not None else None
        
        prompt_builder = get_prompt_builder(self.model_name, eos=self.tokenizer.eos_token, bos=self.tokenizer.bos_token)
        conversation = [
            {"from": "human", "value": "Describe the ongoing action of the claw according to the content of the video."},
            {"from": "gpt", "value": task_description},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        prompt = prompt_builder.get_prompt()

        prefix_prompt = prompt[:prompt.find(task_description)]
        middle_prompt = task_description
        suffix_prompt = prompt[prompt.find(task_description)+len(task_description):]

        prefix_input_ids = self.tokenizer(prefix_prompt, add_special_tokens=False).input_ids
        middle_input_ids = self.tokenizer(middle_prompt, add_special_tokens=False).input_ids
        suffix_input_ids = self.tokenizer(suffix_prompt, add_special_tokens=True).input_ids
        input_ids = prefix_input_ids + middle_input_ids + suffix_input_ids
        labels = [IGNORE_INDEX for _ in range(len(prefix_input_ids))] + middle_input_ids + [IGNORE_INDEX for _ in range(len(suffix_input_ids))]

        if self.tokenizer.eos_token is not None and self.tokenizer.eos_token_id != input_ids[-1]:
            input_ids = input_ids + [self.tokenizer.eos_token_id]
            labels = labels + [IGNORE_INDEX]

        if self.predict_stop_token and self.tokenizer.eos_token_id is not None:
            labels[-1] = self.tokenizer.eos_token_id

        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        attention_mask = torch.ones_like(input_ids)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            video_tensor=video_tensor,
            gripper_video_tensor=gripper_video_tensor
        )

@dataclass
class VideoCaptionPaddedCollator:
    pad_token_id: int

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask, video_tensor, gripper_video_tensor = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "attention_mask", "video_tensor", "gripper_video_tensor")
        )
        input_ids = pad_sequences(input_ids, padding_value=self.pad_token_id)
        labels = pad_sequences(labels, padding_value=IGNORE_INDEX)
        attention_mask = pad_sequences(attention_mask, padding_value=False)

        video_tensor = torch.stack(video_tensor)
        gripper_video_tensor = torch.stack(gripper_video_tensor) if gripper_video_tensor[0] is not None else None

        return {
            "rgb": video_tensor,
            "hand_rgb": gripper_video_tensor,
            "text": input_ids,
            "text_mask": attention_mask,
            "instr_and_action_ids": input_ids,
            "instr_and_action_labels": labels,
            "instr_and_action_mask": attention_mask,
        }

class VideoCaptionDataset(BaseTaskDataset):
    """
    Abstract dataset base class.

    Args:
        num_workers: Number of dataloading workers for this dataset.
        batch_size: Batch size.
    """
    def __init__(
        self,
        model_name: str="flamingo",
        predict_stop_token: bool=True,
        **kwargs
    ):
        kwargs["task_type"] = "video_caption"
        self.model_name, self.predict_stop_token = model_name, predict_stop_token
        super().__init__(**kwargs)
        
    def init_batch_transform(self):
        return VideoCaptionBatchTransform(
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            text_fn=self.text_fn,
            image_fn=self.image_fn,
            predict_stop_token=self.predict_stop_token,
        )
    
    def init_collater_fn(self):
        # use or to avoid the attr exists but the value is None
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        return VideoCaptionPaddedCollator(
            pad_token_id=pad_token_id,
        )
