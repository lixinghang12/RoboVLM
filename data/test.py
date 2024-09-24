import numpy as np
import pickle as pkl
from copy import deepcopy
from data.calvin_dataset import DiskCalvinDataset
from data.calvin_dataset_video import DiskCalvinVideoDataset
from data.calvin_dataset_unified import CalvinDataset

def get_left_pad_index(episode_chunk_mask: np.ndarray, window_size: int, index: int=0, pad_num: int=1):
    for i in range(index, len(episode_chunk_mask)):
        if episode_chunk_mask[i][:window_size].sum() == window_size - pad_num:
            return i
    return -1

def get_right_pad_index(episode_chunk_mask: np.ndarray, window_size: int, index: int=0, pad_num: int=1):
    for i in range(index, len(episode_chunk_mask)):
        if episode_chunk_mask[i][window_size:].sum() == episode_chunk_mask.shape[1] - window_size - pad_num:
            return i
    return -1

def test_train_interleave():
    config_path = "DiskCalvinDataset.pkl"
    with open(config_path, 'rb') as file:
        config = pkl.load(file)
    unified_config = deepcopy(config)
    unified_config['mode']="train"
    unified_config['organize_type'] = "interleave"
    unified_config['discrete'] = True
    unified_config['window_sample'] = 'sliding'
    unified_config["chunk_action"] = True
    unified_config['left_pad'] = False
    dataset = CalvinDataset(**unified_config)

    sample_index1 = get_right_pad_index(dataset.chunk_mask_lookup, dataset.window_size, pad_num=0)
    sample_index2 = get_right_pad_index(dataset.chunk_mask_lookup, dataset.window_size, sample_index1, pad_num=1)
    sample_index3 = get_right_pad_index(dataset.chunk_mask_lookup, dataset.window_size, sample_index2, pad_num=dataset.act_step-1)
    samples = []
    import pdb; pdb.set_trace()
    for i in [sample_index1, sample_index2, sample_index3]:
        samples.append(dataset[i])

    samples = dataset.collater(samples)

test_train_interleave()

config_path2 = "DiskCalvinVideoDataset.pkl"
with open(config_path2, 'rb') as file:
    config2 = pkl.load(file)
unified_config2 = deepcopy(config2)
unified_config2['mode']="train"
unified_config2['organize_type'] = "segment"
unified_config2['discrete'] = True
unified_config2['window_sample'] = 'range'
unified_config2["chunk_action"] = False
unified_config2['left_pad'] = True
dataset2 = DiskCalvinVideoDataset(**config2)
unified_dataset2 = CalvinDataset(**unified_config2)
sample2 = []
unified_sample2 = []
import pdb; pdb.set_trace()
for i in [0, 53, 54]:
    sample2.append(dataset2[i])
    unified_sample2.append(unified_dataset2[i])
    x = sample2[-1]
    y = unified_sample2[-1]
sample2 = dataset2.collater(sample2)
unified_sample2 = unified_dataset2.collater(unified_sample2)
from IPython import embed; embed()
# TODO Need to test if mode=inference what gonna to happen



