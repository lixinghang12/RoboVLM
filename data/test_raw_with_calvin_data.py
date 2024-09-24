import pickle as pkl
from data.calvin_dataset_six_month import DiskCalvinDataset_BAK
from data.calvin_dataset import DiskCalvinDataset
import torch
import tqdm
from pathlib import Path

with open('dataset-training-flamingo.pkl', 'rb') as f:
    dataset_config = pkl.load(f)

dataset_config['use_mu_law'] = False
dataset_config['rgb_pad'] = -1
dataset_config['gripper_pad'] = -1
dataset_config['act_step'] = dataset_config['fwd_pred_next_n']
dataset = DiskCalvinDataset(**dataset_config)
dataset_config['text_fn'] = dataset.text_fn
dataset_config['datasets_dir'] = Path(dataset_config['data_dir'])
dataset_raw = DiskCalvinDataset_BAK(**dataset_config)
key_set = set()

for i in tqdm.tqdm(range(200)):
    import pdb; pdb.set_trace()
    data = dataset.collater([dataset[i]])
    data_raw = dataset_raw.collater([dataset_raw[i]])
    image_tensors, (text_tensors, attention_mask), action_tensors, gripper_tensors, state_tensors, robot_obs, raw_images = data_raw
    if not torch.all(data['rgb'] == image_tensors):
        key_set.add('rgb')
        print('rgb')
    if not torch.all(data['hand_rgb'] == gripper_tensors):
        key_set.add('hand_rgb')
    
    if not torch.all(data['text'] == text_tensors):
        key_set.add('text')
        print('text')
    if not torch.all(data['text_mask'] == attention_mask):
        key_set.add('text_mask')
        print('text_mask')
    action_tensors[..., -1] = (action_tensors[..., -1] + 1) / 2
    if not torch.all(data['action'] == action_tensors):
        if 'action' not in key_set:
            key_set.add('action')
            from IPython import embed; embed()
from IPython import embed; embed()