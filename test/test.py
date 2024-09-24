from pathlib import Path
import os
import random
import numpy as np
import tqdm
import base64
import requests
from PIL import Image
import warnings
import numpy as np
import requests
from image_utils import *
from lang_sam import LangSAM
import torch
import logging
from sam_utils import *
from typing import List, Tuple

root_dir = Path('calvin_debug_dataset/training')
image_dir = root_dir / "images"
os.makedirs(image_dir, exist_ok=True)
file_format = "episode_0{}.npz"

def test():
    language_info = np.load(root_dir / 'lang_annotations' / 'auto_lang_ann.npy', allow_pickle=True)
    language_info = language_info.tolist()
    ann_list = language_info['language']['ann']
    task_list = language_info['language']['task']
    index_list = language_info['info']['indx']
    save_task_set = set()
    model = LangSAM()
    for index_range, ann, task in zip(index_list, ann_list, task_list):
        if task in save_task_set:
            continue
        save_task_set.add(task)
        start_id, end_id = index_range
        indices, begin_status = get_diff_gripper_index_list(start_id, end_id)
        close_begin_index = 1 if begin_status else 0
        
        for i in range(close_begin_index, len(indices)-1, 2):
            close_index = indices[i]
            open_index = indices[i+1]
            print(f"close_index: {close_index}, open_index: {open_index}")
            mask_list, logit_list, box_list, background_mask_list, background_points_list = close2open_target_mask(model, [image_dir / f"{index}.png" for index in range(close_index, open_index)])
            # TODO remove the open2close_target_mask
            # if i != 0:
            #     last_open_index = indices[i-1]
            #     print(f"last_open_index: {last_open_index}, close_index: {close_index}")
            #     open_mask_list, open_logit_list = open2close_target_mask(model, [image_dir / f"{index}.png" for index in range(last_open_index, close_index)], logit_list[0])
            #     mask_list = open_mask_list + mask_list
            #     logit_list = open_logit_list + logit_list
            for j, mask in enumerate(mask_list[::-1]):
                image = Image.open(image_dir / f"{open_index-j-1}.png")
                if len(mask) == 0:
                    continue
                save_image_with_all_elements(image, mask[None], save_path=image_dir / f"{open_index-j-1}-mask.png")
                save_image_with_all_elements(
                    image=image,
                    masks=background_mask_list[j][None], 
                    boxes=box_list[j][None],
                    background_points=background_points_list[j],
                    save_path=image_dir / f"{open_index-j-1}-background-mask.png"
                )

def save_image(index: int):
    data = np.load(root_dir / file_format.format(index))
    image = data['rgb_gripper']
    image = Image.fromarray(image)
    image.save(image_dir / f"{index}.png")

def close2open_target_mask(
    model: LangSAM, image_path_list: List[Path], 
    text_prompt: str = "A cube in the middle of a pair of white jaws",
    background_prompts: str = "A pair white clamping jaws on the machine arm"
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    masks_list = []
    logits_list = []
    scores_list = []
    masks_image_list = []
    background_mask_list = []
    boxes_list = []
    background_points_list = []
    for image_path in tqdm.tqdm(image_path_list):
        image = Image.open(image_path)
        masks, logits, boxes, phrases, scores, background_mask_np, background_points = segement_with_prompt(model, image, text_prompt, background_prompts)
        image_array = np.array(image)
        # repeat the image in the first dim to align with the masks
        image_arrays = image_array[None].repeat(len(masks), axis=0)
        masks_image_list.append(image_arrays*masks[..., None])
        masks_list.append(masks)
        logits_list.append(logits)
        scores_list.append(scores)
        background_mask_list.append(background_mask_np)
        boxes_list.append(boxes)
        background_points_list.append(background_points)
        

    # TODO delete the save cache code for easily debug
    # not_empty_indices = [i for i, masks in enumerate(masks_list) if len(masks) > 0]
    # not_empty_masks_list = [masks_list[i] for i in not_empty_indices]
    # not_empty_masks_image_list = [masks_image_list[i] for i in not_empty_indices]
    # not_empty_scores_list = [scores_list[i] for i in not_empty_indices]
    # save_cache(not_empty_masks_list, 'masks.pkl')
    # save_cache(not_empty_masks_image_list, 'masks_image.pkl')
    # save_cache(not_empty_scores_list, 'scores.pkl')
    # save_cache(background_mask_list, 'background_mask.pkl')
    # save_cache(background_points_list, 'background_points.pkl')
    # save_image_with_all_elements(
    #     masks=background_mask_list[0][None],
    #     background_points=background_points_list[0],
    #     save_path='test.png'
    # )
    # s = SimilarityMatrix(not_empty_masks_image_list, not_empty_scores_list)
    # l = s.get_best_choices()
    # net = Net(not_empty_masks_list)
    # mask_choice_list = net.get_shorest_path()
    mask_choice_list = get_choice_list(masks_image_list, scores_list)
    mask_list = []
    logit_list = []
    box_list = []
    # for mask_choice in mask_choice_list:
    for mask_choice, masks, logits, boxes in zip(mask_choice_list, masks_list, logits_list, boxes_list):
        if mask_choice != -1:
            mask_list.append(masks[mask_choice])
            logit_list.append(masks[mask_choice])
            box_list.append(boxes[mask_choice])
        else:
            mask_list.append(np.zeros(image_array.shape[:2]))
            logit_list.append(np.zeros((256, 256)))
            box_list.append(np.zeros(4))
    return mask_list, logit_list, box_list, background_mask_list, background_points_list

def open2close_target_mask(
    model: LangSAM, image_path_list: List[Path|str], logit: np.ndarray,  bbox: Tuple[int, int, int, int],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """ segement the target object in the frames. The gripper is open in the first frame and close in the last frame.

    Args:
        model (LangSAM): the model to predict the object in the image
        image_path_list (list[Path | str]): the image path list of the frames
        logit (np.ndarray): [256, 256], the target object pixel-logit in the last frame
        bbox (tuple[int, int, int, int]): the bbox of the target object in the last frame

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: _description_
    """
    mask_list = []
    logit_list = []
    for image_path in tqdm.tqdm(image_path_list[::-1]):
        image = Image.open(image_path)
        mask, logit = segement_by_sam(model, image, logit, bbox)
        mask_list.append(mask)
        logit_list.append(logit)
        bbox = get_bbox(mask)
        # save_image_name = image_path.name.split(".")[0] + "-mask.png"
        # save_image_path = image_dir / save_image_name
        # save_image_with_all_elements(image, mask[None], save_path=save_image_path)
    return mask_list[::-1], logit_list[::-1]

def get_diff_gripper_index_list(start_id, end_id) -> Tuple[List[int], bool]:
    """
    return
        change_indices: list[int], the indices of the episode where the gripper status changes
        begin_status: bool, the status of the gripper at the beginning of the episode
    """
    gripper_status_list = []
    for index in tqdm.tqdm(range(start_id, end_id + 1)):
        file_name = file_format.format(index)
        data = np.load(root_dir / file_name)
        gripper_status = data['robot_obs'][-1]
        gripper_status_list.append(gripper_status)

    gripper_status_list = np.array(gripper_status_list)
    gripper_status_list = gripper_status_list > 0
    gripper_status_diff = np.diff(gripper_status_list)
    change_indices = np.where(gripper_status_diff != 0)[0]
    change_indices += 1 + start_id
    change_indices = change_indices.tolist()
    change_indices = [start_id] + change_indices
    if end_id - change_indices[-1] > 1:
        change_indices.append(end_id)
    return change_indices, gripper_status_list[0]

def encode_image(index: int):
    image_path = image_dir / f"{index}.png"
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def construct_message_to_gpt4():
    api_key = "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"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            # {
            #     "role": "system",
            #     "content": "You are a helpful AI assistant. You need to help me split a robot task into smaller subtasks. I will give you the basis for this process."
            # },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": open('prompt.txt', 'r').read()
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(360619)}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(360635)}"
                        }
                    },
                ]
            }
        ],
        "max_tokens": 4000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # print(response.json())
    print(response.json()['choices'][0]['message']['content'])
    with open('response.txt', 'w') as file:
        file.write(str(response.json()))
    from IPython import embed; embed()

# test()
# construct_message_to_gpt4()
if __name__ == "__main__":
    # model = LangSAM()
    # predict_sam_with_background(model, Image.open('/opt/dongwang/RobotVLM/calvin_debug_dataset/training/images/358656.png'))
    test()