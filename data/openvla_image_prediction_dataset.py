from logging.config import dictConfig
from typing import Any, Dict
from data.base_openvla_dataset import RLDSDataset
from data.base_future_frame_prediction_dataset import ImagePredictionDataset

class OpenVLAImagePredictionDataset(ImagePredictionDataset, RLDSDataset):
    def __init__(
        self,
        **kwargs
    ):
        frame_num = kwargs['frame_num']
        kwargs['window_size'] = 1
        kwargs['fwd_pred_next_n'] = frame_num - 1
        kwargs['chunk_action'] = False

        ImagePredictionDataset.__init__(self, **kwargs)        
        RLDSDataset.__init__(self, **kwargs)
    
    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in RLDSDataset.__iter__(self):
            yield self.batch_transform(
                task_description=rlds_batch['task']['language_instruction'].decode(),
                images=rlds_batch['observation']['image_primary'],
                gripper_images=None
            )

if __name__ == "__main__":
    import clip
    from functools import partial
    import torch
    _, clip_preprocess = clip.load('ViT-B/32', device='cpu')
    image_size = 224
    def preprocess_image(sample, image_processor):
        image = [image_processor(s).unsqueeze(0) for s in sample]
        image = torch.cat(image, dim=0)
        # apply random horizontal flip and color jitter
        return image

    def print_shape(data: Dict[str, torch.Tensor]):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {value}")

    config = dict(
        data_root_dir="/mnt/bn/robotics-data-lxh-lq/openvla/datasets/open-x-embodiment",
        # data_mix="bridge",
        data_mix="rt_1",
        image_size=image_size,
        image_fn=partial(preprocess_image, image_processor=clip_preprocess),
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
        model_name="qwen",
        mode="train",
        predict_stop_token=True,
        traj_cons=True,
        shuffle_buffer_size=4,
        train=True,
        rgb_pad=2,
        gripper_pad=2,
        image_aug=True,
        frame_num=10
    )

    dataset = OpenVLAImagePredictionDataset(**config)
    samples = []
    count = 0
    import pdb; pdb.set_trace()
    for sample in dataset.__iter__():
        if count == 4:
            break
        samples.append(sample)
        count += 1

    batch_data = dataset.collater(samples)
    print_shape(batch_data)
    print("pass code test!!!\n")
