import copy

import lightning.pytorch as pl
from lightning.pytorch.utilities.combined_loader import CombinedLoader
import data as Datasets
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler, RandomSampler
from data.weighted_combined_loader import WeightedCombinedLoader


import data.samplers as gr_samplers
from copy import deepcopy
from utils.dist_train import get_rank, is_dist
from utils.utils import collate_with_none
import traceback
import os
from datasets.distributed import split_dataset_by_node


class GRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        batch_size,
        num_workers,
        data_root='',
        **kwargs
    ):
        super().__init__()
        self.train_dataset_config = train_dataset
        self.val_dataset_config = val_dataset
        self._train_datasets = []
        self._val_datasets = []
        self._train_loader = None
        self._val_loader = None
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

    def _check_data_path(self, data_cfg):
        print(self.data_root)
        if data_cfg['type'] == 'ConcatDataset':
            data_cfg['datasets'] = [self._check_data_path(d) for d in data_cfg['datasets']]
        elif 'data_dir' in data_cfg and not os.path.isabs(data_cfg['data_dir']):
            data_cfg['data_dir'] = os.path.join(self.data_root, data_cfg['data_dir'])
        return data_cfg

    def _init_dataset(self, dataset_config, batch_size, num_workers, is_training=True):
        dataset_config = self._check_data_path(dataset_config)

        # avoid modification of the self attributes
        dataset_config = copy.deepcopy(dataset_config)
        dataset_type = dataset_config.pop('type')
        # assert dataset_type in {'ConcatDataset', 'GRDataset', 'DiskCalvinDataset', 'DiskCalvinVideoDataset', 'Real_Dataset', 'VideoLLaVADataset'}
        dataset_config['is_training'] = is_training
        sampler_config = dataset_config.pop('sampler', None)

        dataset_config.update(self.kwargs)
        
        # mode = dataset_config['data_dir'].split('/')[-1]
        # with open(f'dataset-{mode}.pkl', 'wb') as file:
        #     import pickle as pkl
        #     pkl.dump(dataset_config, file)
            
        dataset = getattr(Datasets, dataset_type)(**dataset_config)

        sampler_cls = None
        if sampler_config is not None:
            sampler_type = sampler_config.pop('type')
            sampler_cls = getattr(gr_samplers, sampler_type, None)

        if sampler_cls is not None:
            # FIXME: this is_training is not in every sampler's arg list.
            #   Consider to use inspect package to fix this.
            sampler_config['is_training'] = is_training
            sampler_config['dataset'] = dataset
            sampler = sampler_cls(**sampler_config)
        elif is_dist():
            # default to be distributed sampler
            sampler = DistributedSampler(dataset, shuffle=True, drop_last=False, seed=self.kwargs.get('seed', 123))
        elif is_training:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers if is_training else num_workers // 2,
            sampler=sampler,
            drop_last=True,
            collate_fn=dataset.collater if hasattr(dataset, 'collater') else collate_with_none,
            pin_memory=True
        )

        return dataset, data_loader

    def _init_iterable_dataset(self, dataset_config, batch_size, num_workers, is_training=True):
        dataset_config = self._check_data_path(dataset_config)

        # avoid modification of the self attributes
        dataset_config = copy.deepcopy(dataset_config)

        datset_type = dataset_config.pop('type')
        # assert datset_type in {'ImageTextDataset', 'RTXDataset', 'VideoLLaVADataset'}

        dataset_config.update(self.kwargs)
        dataset_config['is_training'] = is_training
        dataset = getattr(Datasets, datset_type)(**dataset_config)
        # dataset = split_dataset_by_node(dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers if is_training else num_workers // 2,
            drop_last=True,
            collate_fn=dataset.collater if hasattr(dataset, 'collater') else collate_with_none,
            pin_memory=True
        )

        return dataset, data_loader

    def _init_datasets(self, dataset_config, is_training, batch_size, num_workers):
        if isinstance(dataset_config, dict):
            if get_rank() == 0:
                print("=" * 40)
                print("Initializing dataloader from config:")
                for k in dataset_config:
                    print(f"{k}: {dataset_config[k]}")
                print(f"is_training: {is_training}")
                print(f"batch_size: {batch_size}")
                print(f"num_workers: {num_workers}")
            dataset_type = dataset_config['type']
            assert isinstance(batch_size, int)
            assert isinstance(num_workers, int)
            # if dataset_type in {'ImageTextDataset', 'RTXDataset'} or 'OpenVLA' in dataset_type:
            if dataset_type in {'ImageTextDataset', 'RTXDataset'}:
                return self._init_iterable_dataset(
                    dataset_config, is_training=is_training,
                    batch_size=batch_size, num_workers=num_workers)
            else:
                return self._init_dataset(
                    dataset_config, is_training=is_training,
                    batch_size=batch_size, num_workers=num_workers)
        else:
            assert isinstance(dataset_config, list)
            all_sets_and_loaders = []
            assert isinstance(batch_size, (tuple, list)) and len(batch_size) == len(dataset_config)
            assert isinstance(num_workers, (tuple, list)) and len(num_workers) == len(dataset_config)
            for i, config in enumerate(dataset_config):
                all_sets_and_loaders.append(self._init_datasets(
                    config, is_training=is_training, batch_size=batch_size[i],
                    num_workers=num_workers[i]))
            datasets, dataloaders = zip(*all_sets_and_loaders)
            if is_training:
                combined_dataloader = WeightedCombinedLoader(dataloaders, "max_size_cycle", weights=self.kwargs.get("weights", None))
                # combined_dataloader = CombinedLoader(dataloaders, "max_size_cycle")
                return datasets, combined_dataloader
            else:
                return datasets, dataloaders

    def _init_dataset_params(self, is_training, param_name='batch_size'):
        param = getattr(self, param_name)
        if not is_training:
            # setting for val datasets
            if isinstance(param, (tuple, list)):
                if isinstance(self.val_dataset_config, (tuple, list)):
                    param = [param[0]] * len(self.val_dataset_config)
                else:
                    param = param[0]
            else:
                if isinstance(self.val_dataset_config, (tuple, list)):
                    param = [param] * len(self.val_dataset_config)
                else:
                    param = param
        else:
            if isinstance(param, int):
                if isinstance(self.train_dataset_config, (tuple, list)):
                    param = [param] * len(self.train_dataset_config)
            elif isinstance(param, (tuple, list)):
                assert isinstance(self.train_dataset_config, (tuple, list)) and len(self.train_dataset_config) == len(param)
        return param

    def initialize(self, mode='train'):
        if mode == 'train':
            batch_size = self._init_dataset_params(True, 'batch_size')
            num_workers = self._init_dataset_params(True, 'num_workers')
            self._train_datasets, self._train_loader = self._init_datasets(
                self.train_dataset_config, True, batch_size, num_workers)

        elif mode == 'val':
            batch_size = self._init_dataset_params(False, 'batch_size')
            num_workers = self._init_dataset_params(False, 'num_workers')
            self._val_datasets, self._val_loader = self._init_datasets(
                self.val_dataset_config, False, batch_size, num_workers)
            if get_rank() == 0:
                print(f"val_loader size: {len(self._val_loader)}")

    def train_datasets(self):
        return self._train_datasets

    def val_datasets(self):
        return self._val_datasets

    def train_dataloader(self):
        self.initialize('train')
        return self._train_loader

    def val_dataloader(self):
        self.initialize('val')
        return self._val_loader