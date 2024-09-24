import sys
import socket
import struct
import pickle

class NUCClient:
    """
    this client need to read the data from sensors and then send it to trial
    Therefore, this class will maintain two dialogues, one communicates with sensor and the other communicates with trial.
    """
    def __init__(self, server_ip, port) -> None:
        self.HEADER = 64
        self.PORT = port
        self.FORMAT = "utf-8"
        self.DISCONNECT_MSG = "[DISCONNECT SERVICE] ..."
        self.SERVER = server_ip
        self.ADDR = (self.SERVER, self.PORT)
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(self.ADDR)

    def request_info(self, msg_content):
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

    def send_traj(self, traj, timestamp, home=False):
        traj_dict = {'traj': traj, 'timestamp': timestamp, 'home': home}

        data = pickle.dumps(traj_dict)
        size = sys.getsizeof(data)
        header = struct.pack("i", size)

        self.client.sendall(header)
        self.client.sendall(data)
        msg = self.client.recv(2048)
        print(msg)

    def send_msg(self, msg):
        msg_dict = {'msg': msg}

        data = pickle.dumps(msg_dict)
        size = sys.getsizeof(data)
        header = struct.pack("i", size)

        self.client.sendall(header)
        self.client.sendall(data)
        msg = self.client.recv(2048)

        return msg
    
    def send_msg_no_resp(self, msg):
        msg_dict = {'msg': msg}

        data = pickle.dumps(msg_dict)
        size = sys.getsizeof(data)
        header = struct.pack("i", size)

        self.client.sendall(header)
        self.client.sendall(data)
        return


class agent:
    def __init__(self, server_ip, port):
        self.eval_client = NUCClient(server_ip, port)

    def request_info(self, msg):
        data = self.eval_client.request_info("[REQUEST_INFO] --hand_rgb --hand_cam_info --robot_status --static_rgb --static_cam_info --timestamp")
        return data

    def action(self, pose):
        msg = self.eval_client.send_msg(
            f"[SET_EEF_POSE] --pose {str([pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]])}"
        )

    def action_trajectory(self, trajectory, timestamp):
        self.eval_client.send_traj(trajectory, timestamp)

    def action_repeats(self, pose, repeats, timestamp):
        trajectory = [pose for _ in range(repeats)]
        self.eval_client.send_traj(trajectory, timestamp)

    def action_home(self, pose, repeats, timestamp):
        trajectory = [pose for _ in range(repeats)]
        self.eval_client.send_traj(trajectory, timestamp, home=True)

    def close_action(self):
        msg = self.eval_client.send_msg(f"[CLOSE_ACTION]")
        print(f"[SEND MSG] {msg}")

    def close_gripper(self):
        msg = self.eval_client.send_msg(f"[CLOSE_GRIPPER]")
        print(f"[SEND MSG] {msg}")

    def open_gripper(self):
        msg = self.eval_client.send_msg(f"[OPEN_GRIPPER]")
        print(f"[SEND MSG] {msg}")
    
    def image_compare(self, task_id):
        self.eval_client.send_msg_no_resp(f"[COMPARE_TASK_IMG] {task_id}")
