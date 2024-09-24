from ctypes import wstring_at
from distutils.command.config import config
from email.policy import strict
import json
import os.path
from copy import deepcopy
import torch
import torchvision.transforms as T
from PIL import Image
from typing import Literal
import copy
import numpy as np
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from scripts.main import init_trainer_config
from train.flamingo_trainer import FlamingoTrainer
from train.flamingo_video_trainer import FlamingoVideoTrainer
from train.llava_trainer import LLaVATrainer
from train.qwen_trainer import QwenTrainer
from utils.model_utils import build_tokenizer
from data.datamodule.gr_datamodule import GRDataModule
from data.calvin_dataset_video import DiskCalvinVideoDataset
import functools
from data.data_utils import preprocess_image, get_prompt_builder, tcp_to_world_frame
from queue import Queue
from model.policy_head.action_tokenizer import ActionTokenizer

fwd_decay_ratio = 1

class CustomModelLLaVA(CalvinBaseModel):
    # model option
    def __init__(self,
                 ckpt_path,
                 configs,
                 device,
                 save_dir=None,
                 raw_calvin=False,
                 debug=False,
                 action_ensemble=False):
        
        self.model = LLaVATrainer(configs=configs)
        self.init_config(ckpt_path, configs, device, save_dir, raw_calvin, debug)
    
    def process_multi_modal(self, rgb, ):
        pass