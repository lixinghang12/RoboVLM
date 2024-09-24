from dataclasses import dataclass
from tkinter.ttk import LabeledScale
import numpy as np
from typing import Any, Dict, Callable, Optional, Sequence
import torch
from transformers import PreTrainedTokenizerBase

from data.base_task_dataset import BaseTaskDataset
from data.data_utils import get_prompt_builder, get_tensor_chunk, pad_sequences

@dataclass
class ImagePredictionBatchTransform:
    """
    Transform one item of dataset
    """
    model_name: str
    tokenizer: PreTrainedTokenizerBase
    text_fn: Callable
    image_fn: Callable
    window_size: int
    fwd_pred_next_n: int
    predict_stop_token: bool

    def __call__(self, images: np.ndarray, gripper_images: Optional[np.ndarray], task_description: str) -> Dict[str, Any]:
        """Converts an item to the format expected by collator/models."""
        image_tensors = self.image_fn(images)
        image_chunk = get_tensor_chunk(image_tensors, self.fwd_pred_next_n)[1:]
        image_tensors = image_tensors[:self.window_size]
        if gripper_images is not None:
            gripper_image_tensors = self.image_fn(gripper_images, static=False)
            gripper_image_chunk = get_tensor_chunk(gripper_image_tensors, self.fwd_pred_next_n)[1:]
            gripper_image_tensors = gripper_image_tensors[:self.window_size]
        else:
            gripper_image_tensors, gripper_image_chunk = None, None

        prompt_builder = get_prompt_builder(self.model_name, eos=self.tokenizer.eos_token, bos=self.tokenizer.bos_token)
        conversation = [
            {"from": "human", "value": f"Predict the next {self.fwd_pred_next_n} images to {task_description}"},
            {"from": "gpt", "value": ""},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        prompt = prompt_builder.get_prompt()
        input_ids = self.tokenizer(prompt, add_special_tokens=True).input_ids
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.ones_like(input_ids, dtype=bool)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_tensors=image_tensors,
            image_chunk=image_chunk,
            gripper_image_tensors=gripper_image_tensors,
            gripper_image_chunk=gripper_image_chunk
        )

@dataclass
class ImagePredictionPaddedCollator:
    pad_token_id: int
    task_type: str

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        input_ids, attention_mask, image_tensors, image_chunk, gripper_image_tensors, gripper_image_chunk = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "attention_mask", "image_tensors", "image_chunk", "gripper_image_tensors", "gripper_image_chunk")
        )

        input_ids = pad_sequences(input_ids, padding_value=self.pad_token_id)
        attention_mask = pad_sequences(attention_mask, padding_value=False)

        image_tensors = torch.stack(image_tensors)
        gripper_image_tensors = torch.stack(gripper_image_tensors) if gripper_image_tensors[0] is not None else None
        image_chunk = torch.stack(image_chunk)
        gripper_image_chunk = torch.stack(gripper_image_chunk) if gripper_image_chunk[0] is not None else None
        
        return {
            "rgb": image_tensors,
            "hand_rgb": gripper_image_tensors,
            "fwd_rgb_chunck": image_chunk,
            "fwd_hand_rgb_chunck": gripper_image_chunk,
            "fwd_mask": torch.ones(image_chunk.shape[:3], dtype=bool),
            "text": input_ids,
            "text_mask": attention_mask,
            "data_source": self.task_type
        }

class ImagePredictionDataset(BaseTaskDataset):
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
        window_size: int=1,
        fwd_pred_next_n: int=1,
        **kwargs
    ):
        self.model_name, self.predict_stop_token, self.window_size, self.fwd_pred_next_n = model_name, predict_stop_token, window_size, fwd_pred_next_n
        kwargs['task_type'] = "future_frame"
        super().__init__(**kwargs)
        self.init_batch_transform()
        self.init_collater_fn()
        
    def init_batch_transform(self):
        self.batch_transform = ImagePredictionBatchTransform(
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            text_fn=self.text_fn,
            image_fn=self.image_fn,
            predict_stop_token=self.predict_stop_token,
            window_size=self.window_size,
            fwd_pred_next_n=self.fwd_pred_next_n
        )
    
    def init_collater_fn(self):
        # use or to avoid the attr exists but the value is None
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        self.collater = ImagePredictionPaddedCollator(
            pad_token_id=pad_token_id,
            task_type=self.task_type
        )
