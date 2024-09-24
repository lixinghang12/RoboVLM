import argparse
from ast import List
import asyncio
import sys
import pickle
import struct
import threading
import socket
import json
import time
import requests
import struct
import pickle
import numpy as np
import constants as const
from utils import euler2rotm, quat2rotm, rotm2euler, rotm2quat_ros

class TrialClient:
    def __init__(self, save_path: str):
        with open(save_path, 'r') as file:
            data = json.load(file)
        self.url = data['url']
        self.headers = data['headers']
        self.session = requests.Session()

    def request(self, data):
        response = self.session.get(url=self.url, headers=self.headers, json=data)
        return response.json()


class NUCClient:
    def __init__(self, server_ip, port) -> None:
        self.HEADER = 64
        self.PORT = port
        self.FORMAT = "utf-8"
        self.DISCONNECT_MSG = "[DISCONNECT SERVICE] ..."
        self.SERVER = server_ip
        self.ADDR = (self.SERVER, self.PORT)
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(self.ADDR)

    def _request_info(self, msg_content):
        msg_dict = {'msg': msg_content}

        data = pickle.dumps(msg_dict)
        size = sys.getsizeof(data)
        header = struct.pack("i", size)

        self.client.sendall(header)
        self.client.sendall(data)

        header = self.client.recv(4)
        data_size = struct.unpack("i", header)

        data = b""
        while True:
            packet = self.client.recv(4096)
            if packet is not None:
                data += packet
            if packet is None or sys.getsizeof(data) >= data_size[0]:
                break

        data = pickle.loads(data)
        return data

    def _send_traj(self, traj, timestamp, home=False):
        traj_dict = {'traj': traj, 'timestamp': timestamp, 'home': home}

        data = pickle.dumps(traj_dict)
        size = sys.getsizeof(data)
        header = struct.pack("i", size)

        self.client.sendall(header)
        self.client.sendall(data)
        msg = self.client.recv(2048)
        print(msg)

    def _send_msg(self, msg):
        msg_dict = {'msg': msg}

        data = pickle.dumps(msg_dict)
        size = sys.getsizeof(data)
        header = struct.pack("i", size)

        self.client.sendall(header)
        self.client.sendall(data)
        msg = self.client.recv(2048)

        return msg
    
    def _send_msg_no_resp(self, msg):
        msg_dict = {'msg': msg}

        data = pickle.dumps(msg_dict)
        size = sys.getsizeof(data)
        header = struct.pack("i", size)

        self.client.sendall(header)
        self.client.sendall(data)
        return

    def get_status(self):
        data = self._request_info("[REQUEST_INFO] --hand_rgb --hand_cam_info --robot_status --static_rgb --static_cam_info --timestamp")
        return data

    def take_action(self, pose):
        self._send_msg(
            f"[SET_EEF_POSE] --pose {str([pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]])}"
        )

    def take_action_trajectory(self, trajectory, timestamp):
        self._send_traj(trajectory, timestamp)

    def repeat_actions(self, pose, repeats, timestamp):
        trajectory = [pose for _ in range(repeats)]
        self._send_traj(trajectory, timestamp)

    def action_home(self, pose, repeats, timestamp):
        trajectory = [pose for _ in range(repeats)]
        self._send_traj(trajectory, timestamp, home=True)

    def close_action(self):
        msg = self._send_msg(f"[CLOSE_ACTION]")
        print(f"[SEND MSG] {msg}")

    def close_gripper(self):
        msg = self._send_msg(f"[CLOSE_GRIPPER]")
        print(f"[SEND MSG] {msg}")

    def open_gripper(self):
        msg = self._send_msg(f"[OPEN_GRIPPER]")
        print(f"[SEND MSG] {msg}")
    
    def image_compare(self, task_id):
        self._send_msg_no_resp(f"[COMPARE_TASK_IMG] {task_id}")


class Rollout:
    def __init__(self, server_ip: str, port: int, save_trial_parser_har_path: str):
        self.nuc_client = NUCClient(server_ip, port)
        self.trial_client = TrialClient(save_trial_parser_har_path)
        self.nuc_client.close_action()
    
        self.max_iter = const.MAX_ITERS
        self.hand_offset_rotm = const.HAND_OFFSET_ROTM
        self.hand_offset_pos = const.HAND_OFFSET_POS

        self.state_dim = 7
        self.act_dim = 7

        self.is_running = False
        self.desired_fps = const.DESIRED_FPS
        self.save_test_data = const.SAVE_TEST_IMAGE
        self.lock = threading.Lock()
        
        self.reset()
        self._start_video_buffering()

    def reset(self):
        self.iter = 0
        self.data_iter = 0
        self.last_static_rgb = None
        self.last_hand_rgb = None

    def reset_pose(self, init_trans, init_quat):
        init_trans = np.array(init_trans)
        init_trans = init_trans.tolist()
        with self.lock:
            self.nuc_client.take_action(init_trans + init_quat)

    def _start_video_buffering(self):
        print("start video buffering")
        def buffering_thread():
            async def loop():
                while True:
                    task1 = asyncio.create_task(asyncio.sleep(1.0 / self.desired_fps))
                    task2 = asyncio.create_task(self.get_frame())
                    await asyncio.gather(task1, task2)
            asyncio.run(loop())
        
        thread = threading.Thread(target=buffering_thread)
        thread.daemon = True
        thread.start()
    
    async def get_frame(self):
        if self.is_running:
            with self.lock:
                robot_data = self.nuc_client._request_info("[REQUEST_INFO] --robot_status --static_rgb --hand_rgb")
                # convert BGR -> RGB
                self.last_static_rgb = robot_data["static_rgb"][..., ::-1]
                self.last_hand_rgb = robot_data["hand_rgb"][..., ::-1]
                self.last_robot_status = robot_data["robot_status"]
            self.data_iter += 1
    
    def rollout(self, eval_idx, text, init_trans, init_quat, gripper_change_terminate):
        self.is_running = True
        with self.lock:
            self.nuc_client.close_action()
        
        curr_gripper = None
        if init_trans is not None:
            with self.lock:
                self.nuc_client.open_gripper()
            curr_gripper = const.GRIPPER_OPEN
            time.sleep(1.0)

            self.reset_pose(init_trans, init_quat)
            time.sleep(2.0)
            self.nuc_client.compare_image(eval_idx)
            input("Image Alignment!!!")
        
        human_input = input("Rollout (Start: enter, Stop: n)")
        if human_input == "n":
            return
        import pdb; pdb.set_trace()
        while self.iter < self.max_iter:
            # waiting and get the data
            with self.lock:
                robot_status = self.last_robot_status
            curr_eef_xyz_map = robot_status[10:13]
            wait_iter = 0

            with self.lock:
                robot_status = self.last_robot_status
                static_rgb = self.last_static_rgb
                hand_rgb = self.last_hand_rgb
                
            curr_eef_xyz_map = robot_status[10:13]
            curr_eef_quat_map = robot_status[13:17]
            curr_eef_rotm_map = quat2rotm(curr_eef_quat_map)
            gripper_pos = robot_status[44]

            # convert state to hand frame
            curr_hand_xyz_map = curr_eef_xyz_map + np.dot(curr_eef_rotm_map, self.hand_offset_pos)
            curr_hand_rotm_map = curr_eef_rotm_map @ self.hand_offset_rotm
            if curr_gripper is None:
                curr_gripper = const.GRIPPER_CLOSE if gripper_pos > const.GRIPPER_THRESHOLD else const.GRIPPER_OPEN
            robot_state = (curr_hand_xyz_map, curr_hand_rotm_map, curr_gripper)
            if curr_gripper is None:
                curr_gripper = const.GRIPPER_CLOSE if gripper_pos > const.GRIPPER_THRESHOLD else const.GRIPPER_OPEN
            
            action_chunk = self.trial_client.request(dict(
                robot_status=robot_status,
                static_rgb=static_rgb,
                hand_rgb=hand_rgb,
                text=text,
                reset=self.iter==0
            ))
            action_chunk = np.ndarray(action_chunk)
            act_len = action_chunk.shape[0]
            xyz_action_chunk = action_chunk[..., :3]
            rpy_action_chunk = action_chunk[..., 3:6]
            gripper_action_chunk = action_chunk[..., -1]
            
            trajectory_map = []
            for i in range(act_len):
                temp_hand_xyz_map = xyz_action_chunk[i]
                temp_hand_rotm_map = euler2rotm(rpy_action_chunk[i])
                temp_eef_xyz_map = temp_hand_xyz_map - np.dot(temp_hand_rotm_map @ self.hand_offset_rotm.T, self.hand_offset_pos)
                temp_eef_rotm_map = temp_hand_rotm_map @ self.hand_offset_rotm.T
                temp_eef_quat_map = rotm2quat_ros(temp_eef_rotm_map)
                target_eef_pose = np.zeros(7)
                target_eef_pose[:3] = temp_eef_xyz_map
                target_eef_pose[3:] = temp_eef_quat_map
                trajectory_map.append(target_eef_pose)
            
            for target_eef_pose_map, target_gripper in zip(trajectory_map):
                with self.lock:
                    self.nuc_client.take_action(target_eef_pose_map)
                if target_gripper != curr_gripper:
                    gripper_change = True
                    time.sleep(1.0)
                    if target_gripper == const.GRIPPER_CLOSE:
                        with self.lock:
                            self.nuc_client.close_gripper()
                        curr_gripper = const.GRIPPER_CLOSE
                        time.sleep(1.0)
                        break
                    elif target_gripper == const.GRIPPER_OPEN:
                        with self.lock:
                            self.nuc_client.open_gripper()
                        curr_gripper = const.GRIPPER_OPEN
                        time.sleep(1.0)
                        break
                    else:
                        raise ValueError(f"target gripper: {target_gripper}")
                time.sleep(0.1)
            time.sleep(1.0)
            
            # check the action excuted successfully
            target_eef_xyz_map = trajectory_map[-1][:3]
            trans_diff = np.linalg.norm(curr_eef_xyz_map - target_eef_xyz_map)
            while (trans_diff > 0.01) and (wait_iter < 3):
                with self.lock:
                    robot_status = self.last_robot_status
                curr_eef_xyz_map = robot_status[10:13]
                trans_diff = np.linalg.norm(curr_eef_xyz_map - target_eef_xyz_map)
                wait_iter += 1
            
            self.iter += 1
            if gripper_change and gripper_change_terminate:
                break
        
        self.is_running = False
        print("Task done...")
        self.reset()
        sys.exit()
        

def main():
    # hparams
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, help="")
    parser.add_argument("--port", type=int, help="")
    parser.add_argument("--save_trial_parser_har_path", type=str, help="")
    parser.add_argument("--eval_idx", type=int, help="the eval task id")
    parser.add_argument("--text", type=str, help="input text")
    parser.add_argument("--init_trans", help="the init position of the eef, you can type like [1.0 2.1 2.2]")
    parser.add_argument("--init_quat", help="the init quat of the eef, you can type like [1.0 2.1 2.2 3.3]")
    parser.add_argument("--gripper_change_terminate", action='store_true', help="debug mode")
    args = parser.parse_args()
    
    eval_idx = args.eval_idx
    text = args.text
    gripper_change_terminate = args.gripper_change_terminate
    if args.init_trans:
        init_trans = args.init_trans.split('[')[-1].split(']')[0].split(' ')
        init_trans = [float(ele) for ele in init_trans]
        assert len(init_trans) == 3
    else:
        init_trans = None
    
    if args.init_quat:
        init_quat = args.init_quat.split('[')[-1].split(']')[0].split(' ')
        init_quat = [float(ele) for ele in init_quat]
        assert len(init_quat) == 4
    else:
        init_quat = None
        
    RO = Rollout(args.server_ip, args.port, args.save_trial_parser_har_path)
    RO.rollout(eval_idx, text, init_trans, init_quat, gripper_change_terminate)


if __name__ == "__main__":
    main()
    