      
import json
import os
import random
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import json
import math
import warnings
import traceback
def alpha2rotm(a):
    """Alpha euler angle to rotation matrix."""
    rotm = np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ])
    return rotm


def beta2rotm(b):
    """Beta euler angle to rotation matrix."""
    rotm = np.array([
        [np.cos(b), 0, np.sin(b)],
        [0, 1, 0],
        [-np.sin(b), 0, np.cos(b)]
    ])
    return rotm


def gamma2rotm(c):
    """Gamma euler angle to rotation matrix."""
    rotm = np.array([
        [np.cos(c), -np.sin(c), 0],
        [np.sin(c),  np.cos(c), 0],
        [0, 0, 1]
    ])
    return rotm


def euler2rotm(euler_angles):
    """Euler angle (ZYX) to rotation matrix."""
    alpha = euler_angles[0]
    beta = euler_angles[1]
    gamma = euler_angles[2]

    rotm_a = alpha2rotm(alpha)
    rotm_b = beta2rotm(beta)
    rotm_c = gamma2rotm(gamma)

    rotm = rotm_c @ rotm_b @ rotm_a

    return rotm


def isRotm(R):
    # Checks if a matrix is a valid rotation matrix.
    # Forked from Andy Zeng
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotm2euler(R):
    # Forked from: https://learnopencv.com/rotation-matrix-to-euler-angles/
    # R = Rz * Ry * Rx
    assert(isRotm(R))
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
 
    if not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    
    # (-pi , pi]
    while x > np.pi:
        x -= (2 * np.pi)
    while x <= -np.pi:
        x += (2 * np.pi)
    while y > np.pi:
        y -= (2 * np.pi)
    while y <= -np.pi:
        y += (2 * np.pi)
    while z > np.pi:
        z -= (2 * np.pi)
    while z <= -np.pi:
        z += (2 * np.pi)
    return np.array([x, y, z])


class RandomShiftsSingleAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w, f"h, w: {h}, {w}"
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(1, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift = shift.repeat(n, 1, 1, 1)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Easy_Real_Dataset(Dataset):
    def __init__(
            self,
            data_dir,
            seq_len=10,
            act_len=15,
            is_training=True,
            c_act_scaler=[100, 100, 100, 50, 50, 50],
            accumulate_traj_action=False,
            sequence_interval=1,
            shift_padding=[10, 10],
            forward_n_max=7,
    ):
        """Constructor."""
        super().__init__()
        self.forward_n_max=forward_n_max
        self.shift_padding = shift_padding
        self.accumulate_traj_action = accumulate_traj_action
        self.sequence_interval = sequence_interval
        if isinstance(self.shift_padding, int):
            self.shift_padding = (self.shift_padding, self.shift_padding)
        self.seq_len = seq_len
        self.act_len = act_len
        self.mode = 'train' if is_training else 'val'
        self.dataset_dir = os.path.join(data_dir,self.mode)
        self.cache_dir=os.path.join(data_dir,f"seqlen_{self.seq_len}_{self.mode}.json")
        self.c_act_scaler = c_act_scaler
        if isinstance(c_act_scaler, (int, float)):
            self.c_act_scaler = [self.c_act_scaler] * 6
        self.c_act_scaler = np.array(self.c_act_scaler, dtype=float)
        self.action_dim = 7  # ee xyz (3) + ee euler (3) + gripper(1)
        self.state_dim = 7  # ee xyz (3) + ee euler (3) + gripper(1)

        # init preprocessor
        self.static_preprocess, self.hand_preprocess = self._init_preprocess()

        # init annotations and samples
        self.ann_files = self._init_anns(self.dataset_dir)[:100]

        self.samples = self._init_sequences(self.ann_files, self.sequence_interval)
        
        print(f'{len(self.ann_files)} trajectories in total')
        print(f'{len(self.samples)} samples in total')

    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.dataset_dir}"

    def _init_preprocess(self):
        self.input_size = (224, 224)
        self.clip_mean = (0.485, 0.456, 0.406)
        self.clip_std = (0.229, 0.224, 0.225)
        
        if self.mode == 'train':
            static_preprocess = T.Compose([
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                RandomShiftsSingleAug(pad=self.shift_padding[0]), 
                T.Normalize(self.clip_mean, self.clip_std)])
            hand_preprocess = T.Compose([
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                RandomShiftsSingleAug(pad=self.shift_padding[1]), 
                T.Normalize(self.clip_mean, self.clip_std)])
        else:
            static_preprocess = T.Compose([
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
            hand_preprocess = T.Compose([
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])

        return static_preprocess, hand_preprocess

    def _init_anns(self, data_dir):
        ann_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
        return ann_files
    


    def _init_sequences(self, ann_files, sequence_interval):
        if os.path.exists(self.cache_dir):
            with open(self.cache_dir, 'r') as f:
                samples = json.load(f)
        else:
            samples = []
            for ann_file in tqdm(ann_files):
                with open(ann_file, "r") as f:
                    ann = json.load(f)
                n_frames = len(ann['state'])

                # Add start and end
                start = ann['videos'][0]['start']
                end = ann['videos'][0]['end']
                if len(ann['videos']) > 1:
                    for i in range(1, len(ann['videos'])):
                        assert ann['videos'][i]['start'] == start
                        assert ann['videos'][i]['end'] == end
                assert (end - start) == n_frames
                for frame_i in range(n_frames):
                    sample = dict()
                    sample['ann_file'] = ann_file
                    sample['video_frame_ids'] = []
                    sample['frame_ids'] = []
                    curr_frame_i = frame_i
                    while True:
                        if curr_frame_i >= (n_frames - 1):
                            # last frame has no actions
                            break
                        sample['frame_ids'].append(curr_frame_i)
                        sample['video_frame_ids'].append(curr_frame_i + start)# 和frame id之间有何区别？
                        
                        if len(sample['frame_ids']) == self.seq_len:
                            break
                        curr_frame_i += sequence_interval
                    sample["end"]=end
                    sample["start"]=start
                    # make sure there are seq_len number of frames
                    if len(sample['frame_ids']) == self.seq_len:
                        samples.append(sample)
                        # 保存为 JSON 文件
            # with open(self.cache_dir, 'w') as f:
            #     json.dump(samples, f, indent=4)
        return samples

    def __len__(self):
        return len(self.samples)


    def _get_action_sequence(self, label, frame_ids):
        n_frames = len(label['state'])
        import pdb; pdb.set_trace()
        actions = label['action']
        action_seqs = []
        action_masks = []
        for frame_id in frame_ids:
            action_seq = []
            action_mask = np.zeros(self.act_len, dtype=int)
            for curr_frame_id in range(frame_id, frame_id + self.act_len):
                if curr_frame_id >= n_frames - 1:
                    # last frame has no actions
                    break
                action_seq.append(actions[curr_frame_id])
            temp_act_len = len(action_seq)
            action_mask[:temp_act_len] = 1
            if temp_act_len < self.act_len:
                for i in range(self.act_len - temp_act_len):
                    action_seq.append(np.zeros(self.action_dim, dtype=float))
            assert len(action_seq) == self.act_len
            action_seqs.append(action_seq)
            action_masks.append(action_mask)
        action_seqs = np.array(action_seqs)  # (seq_len, act_len, act_dim)
        action_masks = np.array(action_masks)  # (seq_len, act_len)

        if self.accumulate_traj_action:
            accumulate_action_seqs = np.zeros_like(action_seqs)
            accumulate_action_seqs[:, 0] = action_seqs[:, 0]
            accumulate_action_seqs[:, :, -1] = action_seqs[:, :, -1]
            for seq_i in range(self.seq_len):
                curr_action = action_seqs[seq_i, 0]
                curr_xyz = curr_action[0:3]
                curr_rpy = curr_action[3:6]
                curr_rotm = euler2rotm(curr_rpy)
                for act_i in range(1, self.act_len):
                    if action_masks[seq_i, act_i] == 1:
                        rel_action = action_seqs[seq_i, act_i]
                        rel_xyz = rel_action[0:3]
                        rel_rpy = rel_action[3:6]
                        rel_rotm = euler2rotm(rel_rpy)
                        gripper_action = rel_action[-1]
                        curr_xyz = curr_rotm @ rel_xyz + curr_xyz
                        curr_rotm = curr_rotm @ rel_rotm
                        curr_rpy = rotm2euler(curr_rotm)
                        accumulate_action_seqs[seq_i, act_i, 0:3] = curr_xyz
                        accumulate_action_seqs[seq_i, act_i, 3:6] = curr_rpy
                        accumulate_action_seqs[seq_i, act_i, -1] = gripper_action
            action_seqs = accumulate_action_seqs

        c_action_dim = len(self.c_act_scaler)
        action_seqs[:, :, :c_action_dim] *= self.c_act_scaler
        return torch.from_numpy(action_seqs), action_masks
    



    def _load_video_decord(self, video_path, frame_ids):
        """Load video content using Decord"""
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        return frame_data
    
    def _get_frames(self, video_info, frame_ids, preprocess):
        video_path = video_info['video_path']
        replace_prefix=r"/mnt/bn/robotics-data-lxh-lq/real_data/media/0419_easy/"
        remain_path="/".join(video_path.split("/")[-3:])
        video_path=os.path.join(replace_prefix,remain_path)
        crop = video_info.get('crop', None)

        frames = self._load_video_decord(video_path, frame_ids)
        if crop is not None:
            frames = frames[:, crop[0][0]:crop[1][0], crop[0][1]:crop[1][1]]
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2) # (b, c, h, w)
        frames = (frames / 255).float()
        frames = preprocess(frames)
        
        return frames

    def _get_obs(self, label, frame_ids):
        obs = [None, None]
        videos = label['videos']
        for video_i, video in enumerate(videos):
            if video_i == 0:
                frames = self._get_frames(video, frame_ids, self.static_preprocess)
            elif video_i == 1:
                frames = self._get_frames(video, frame_ids, self.hand_preprocess)
            obs[video_i] = frames
        return obs[0], obs[1]

    def _get_relative_states(self, label, frame_ids):
        states = label['state']
        first_id = frame_ids[0]
        first_xyz = np.array(states[first_id][0:3])
        first_rpy = np.array(states[first_id][3:6])
        first_rotm = euler2rotm(first_rpy)
        first_gripper = states[first_id][6]
        first_state = np.zeros(7, dtype=np.float32)
        first_state[-1] = first_gripper
        rel_states = [first_state]
        for k in range(1, len(frame_ids)):
            curr_frame_id = frame_ids[k]
            curr_xyz = np.array(states[curr_frame_id][0:3])
            curr_rpy = np.array(states[curr_frame_id][3:6])
            curr_rotm = euler2rotm(curr_rpy)
            curr_rel_rotm = first_rotm.T @ curr_rotm
            curr_rel_rpy = rotm2euler(curr_rel_rotm)
            curr_rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
            curr_gripper = states[curr_frame_id][6]
            curr_state = np.zeros(7, dtype=np.float32)
            curr_state[0:3] = curr_rel_xyz
            curr_state[3:6] = curr_rel_rpy
            curr_state[-1] = curr_gripper
            rel_states.append(curr_state)
        return torch.from_numpy(np.array(rel_states))

    def __getitem__(self, index):
        try:
            if self.mode == 'val':
                np.random.seed(index)
                random.seed(index)
            sample = self.samples[index]
            ann_file = sample['ann_file']
            frame_ids = sample['frame_ids']
            video_frame_ids = sample['video_frame_ids']
            video_start=sample["start"]
            video_end=sample["end"]
            with open(ann_file, "r") as f:
                label = json.load(f)

            # get goal image
            goal_min_idx = video_frame_ids[-1]
            goal_max_idx = min(video_end, video_frame_ids[-1] + self.forward_n_max * self.sequence_interval)
            goal_ids = list(range(goal_min_idx, goal_max_idx))
            goal_idx = random.choice(goal_ids)
            assert (goal_idx >= video_frame_ids[-1]) and (goal_idx <= video_end)

            video_frame_ids.append(goal_idx)




            static_rgbs, hand_rgbs = self._get_obs(label, video_frame_ids)

            goal_rgb=static_rgbs[-1]

            static_rgbs=static_rgbs[:-1]

            hand_rgbs=hand_rgbs[:-1]
            video_frame_ids=video_frame_ids[:-1]

            actions, action_mask = self._get_action_sequence(label, frame_ids) #？？？？
            rel_states = self._get_relative_states(label, frame_ids)#？？？

            tlen = len(static_rgbs)
            is_hand_available = len(hand_rgbs) > 0
            if is_hand_available: assert tlen == len(hand_rgbs)

            _, C, H, W = static_rgbs.shape
            padded_static_rgbs = torch.zeros((self.seq_len, C, H, W)).float()  # (len, C, H, W)
            padded_hand_rgbs = torch.zeros((self.seq_len, C, H, W)).float()  # (len, C, H, W)
            padded_actions = torch.zeros(self.seq_len, self.act_len, self.action_dim).float()  # (len, act_len, act_dim)
            padded_rel_states = torch.zeros(self.seq_len, self.state_dim).float()  # (len, state_dim)

            padded_static_rgbs[:tlen] = static_rgbs
            if is_hand_available: 
                padded_hand_rgbs[:tlen] = hand_rgbs
            padded_actions[:tlen] = actions
            padded_rel_states[:tlen] = rel_states
            assert action_mask.shape == (self.seq_len, self.act_len)
            attention_mask = np.ones(self.seq_len, dtype=np.int32)  # (len,)
            attention_mask[tlen:] = 0

            rgb_data = padded_static_rgbs
            hand_rgb_data = padded_hand_rgbs

            goal_rgb_data = goal_rgb.float() # (C, H, W)

            action_data = padded_actions
            rel_state_data = padded_rel_states
            attention_mask_data = torch.from_numpy(attention_mask).long()
            action_mask_data = torch.from_numpy(action_mask).long()

            #progress
            progress_data=torch.zeros(self.seq_len).float()

            for i in range(len(video_frame_ids)):
                index=video_frame_ids[i]
                progress=(index-video_start)/(video_end-video_start)
                progress_data[i]=progress


            assert torch.sum(attention_mask_data) >= 2
            data = dict()
            data['rgb'] = rgb_data
            data['goal_rgb']=goal_rgb_data
            data['hand_rgb'] = hand_rgb_data
            data['text'] = label['texts']
            data['action'] = action_data
            data['rel_state'] = rel_state_data
            data['attention_mask'] = attention_mask_data
            data['action_mask'] = action_mask_data
            data["progress"]=progress_data
            return data

        except:
            warnings.warn(f"Invalid data encountered: {self.samples[index]}. Skipped "
                            f"(by randomly sampling another sample in the same dataset).")
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            return self[np.random.randint(len(self.samples))]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    seq_len=8
    # data_dir = "/mnt/bn/robotics-data-wht-yg/lpy/real_data/anns/0419_easy"
    data_dir = "/mnt/bn/robotics-data-lxh-lq/real_data/anns/0419_easy/"
    dataset = Easy_Real_Dataset(
        data_dir,
        seq_len=seq_len,
        act_len=15,
        is_training=True,
        c_act_scaler=[100, 100, 100, 50.0, 50.0, 50.0],
        sequence_interval=5,
        shift_padding=[10, 10],
        forward_n_max=7,
        accumulate_traj_action=True
    )

    rgb_mean = np.array([0.485, 0.456, 0.406])
    rgb_std  = np.array([0.229, 0.224, 0.225])
    gripper_set=set()
    for i in range(0, len(dataset),100):
        data = dataset[i]
        text = data['text']
        goal_rgb = data['goal_rgb']
        rgb = data['rgb']
        hand_rgb = data['hand_rgb']
        rel_state = data['rel_state']
        action = data['action']
        progress=data["progress"]
        

        print(f"{text}")
        fig, ax = plt.subplots(seq_len // 3 + 1, 3)
        for k in range(seq_len):
            temp_rgb = rgb[k].permute(1, 2, 0).numpy()
            temp_rgb = temp_rgb * rgb_std + rgb_mean
            temp_rgb = np.clip(temp_rgb, 0.0, 1.0)
            ax[k // 3, k % 3].imshow(temp_rgb)
        temp_rgb = goal_rgb.permute(1, 2, 0).numpy()
        temp_rgb = temp_rgb * rgb_std + rgb_mean
        temp_rgb = np.clip(temp_rgb, 0.0, 1.0)
        ax[seq_len // 3, 2].imshow(temp_rgb)
        plt.savefig("debug_goal_rgb.png", dpi=300)

        fig, ax = plt.subplots(seq_len // 3 + 1, 3)
        for k in range(seq_len):
            temp_rgb = hand_rgb[k].permute(1, 2, 0).numpy()
            temp_rgb = temp_rgb * rgb_std + rgb_mean
            temp_rgb = np.clip(temp_rgb, 0.0, 1.0)
            ax[k // 3, k % 3].imshow(temp_rgb)
        plt.savefig("debug_goal_hand_rgb.png", dpi=300)
        