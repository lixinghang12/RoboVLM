import torch
from typing import Dict
from functools import partial
from data.base_video_caption_dataset import VideoCaptionDataset
from data.base_calvin_dataset import CalvinGlobalDataset


class CalvinVideoCaptionDataset(VideoCaptionDataset, CalvinGlobalDataset):
    def __init__(
        self,
        **kwargs
    ):
        VideoCaptionDataset.__init__(self, **kwargs)
        CalvinGlobalDataset.__init__(self, **kwargs)

    def __getitem__(self, idx: int) -> Dict:
        episode = self._load_episode(idx)
        static_images, gripper_images, lang_ann = (
            episode[key]
            for key in ("static_images", "gripper_images", "lang_ann")
        )
        return self.batch_transform(video=static_images, gripper_video=gripper_images, task_description=lang_ann)

    def __len__(self) -> int:
        return len(self.frame_idx_array)


if __name__ == "__main__":
    import clip
    _, clip_preprocess = clip.load('ViT-B/32', device='cpu')
    image_size = 224
    def preprocess_image(sample, image_processor):
        image = [image_processor(s).unsqueeze(0) for s in sample]
        image = torch.cat(image, dim=0)
        return image

    config = dict(
        data_dir='/mnt/bn/robotics/manipulation_data/calvin_data/task_ABCD_D/training',
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
        rgb_pad=2,
        gripper_pad=2,
        frame_num=20,
    )
    import pdb; pdb.set_trace()
    dataset = CalvinVideoCaptionDataset(**config)
    samples = [dataset[0], dataset[1]]
    batch = dataset.collater(samples)
    from IPython import embed; embed()