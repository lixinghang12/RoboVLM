import os
import sys
import time
import pickle
import struct
import cv2
import rospy
import numpy as np
import socketserver
from cv_bridge import CvBridge
from byte_msgs.msg import EvalSensors
from gripper import Gripper
from robot_client_home import RobotClient as agent
from socket_config import socket_conf
from headcam_compare import ImageCompare


POSE = 0
TRAJECTORY = 1
DEFAULT_ACT_LEN = 15  # TODO: Hardcode

bridge = CvBridge()
CTRL_WHOLE_BODY = 0
CTRL_LOCK_BASE = 1
CTRL_STOP = 4

GRIPPER_NO_ACTION = -1
GRIPPER_OPEN = 2
GRIPPER_CLOSE = 3

RWBC_POS_CTRL = 0
RWBC_VEL_CTRL = 1


class TaskSocketServer(socketserver.StreamRequestHandler):
    def init(self, mode=POSE):
        rospy.init_node("eval_server", anonymous=True)
        self.mode = mode
        self.robot = agent(rate=10, mode=self.mode)

        self.traj_meta = {"traj": [], "timestamp": rospy.Time.now(), "home": False}

        self.gripper = Gripper()

    def process_robot_status(self, msg):
        robot_status = np.zeros(45)
        robot_status[0] = msg.base_pose.x
        robot_status[1] = msg.base_pose.y
        robot_status[2] = msg.base_pose.theta
        robot_status[3] = msg.head_pose.position.x
        robot_status[4] = msg.head_pose.position.y
        robot_status[5] = msg.head_pose.position.z
        robot_status[6] = msg.head_pose.orientation.x
        robot_status[7] = msg.head_pose.orientation.y
        robot_status[8] = msg.head_pose.orientation.z
        robot_status[9] = msg.head_pose.orientation.w
        robot_status[10] = msg.hand_pose.position.x
        robot_status[11] = msg.hand_pose.position.y
        robot_status[12] = msg.hand_pose.position.z
        robot_status[13] = msg.hand_pose.orientation.x
        robot_status[14] = msg.hand_pose.orientation.y
        robot_status[15] = msg.hand_pose.orientation.z
        robot_status[16] = msg.hand_pose.orientation.w
        robot_status[17] = msg.height
        robot_status[18:31] = msg.q
        robot_status[31:44] = msg.qd
        robot_status[44] = msg.gripper_status[0]
        return robot_status

    def handle(self):
        print(f"[{self.client_address}] NEW CONNECTION connected.")

        connected = True

        while connected == True:
            traj_flag = False
            msg_flag = False

            header = self.request.recv(4)
            try:
                size = struct.unpack("i", header)
                data = self.request.recv(size[0])
            except Exception:
                pass

            data = pickle.loads(data)
            if "msg" in data.keys():
                msg = data["msg"]
                msg_flag = True
                print(msg)
            else:
                self.traj_meta["traj"] = data["traj"]
                if data["timestamp"] != None:
                    self.traj_meta["timestamp"] = rospy.Time(
                        data["timestamp"][0], data["timestamp"][1]
                    )
                else:
                    self.traj_meta["timestamp"] = rospy.Time.now()

                # check home
                if "home" in data.keys():
                    self.traj_meta["home"] = True
                traj_flag = True
                print(self.traj_meta["traj"])

            if msg_flag:
                if msg.startswith("[REQUEST_INFO]"):
                    print(f"[{self.client_address}] {msg}.")
                    sensor_info = rospy.wait_for_message("/eval_sensors", EvalSensors)

                    result = {}
                    if "--hand_rgb" in msg:
                        rgb = bridge.imgmsg_to_cv2(
                            sensor_info.rgb[0], sensor_info.rgb[0].encoding
                        )
                        result["hand_rgb"] = rgb
                    if "--hand_depth" in msg:
                        depth = bridge.imgmsg_to_cv2(sensor_info.depth[0], "32FC1")
                        print(depth.shape)
                        result["hand_depth"] = depth
                    if "--hand_cam_info" in msg:
                        result["hand_cam_info"] = np.array(sensor_info.camera_info[0].K)
                    if "--static_rgb" in msg:
                        rgb = bridge.imgmsg_to_cv2(
                            sensor_info.rgb[1], sensor_info.rgb[1].encoding
                        )[:, :, :3]
                        result["static_rgb"] = rgb
                    if "--static_depth" in msg:
                        depth = bridge.imgmsg_to_cv2(sensor_info.depth[1], "16UC1")
                        print(depth.shape)
                        result["static_depth"] = depth
                    if "--static_cam_info" in msg:
                        result["static_cam_info"] = np.array(
                            sensor_info.camera_info[1].K
                        )
                    if "--robot_status" in msg:
                        robot_status = self.process_robot_status(
                            sensor_info.robot_status
                        )
                        result["robot_status"] = robot_status
                    if "--timestamp" in msg:
                        result["timestamp"] = [
                            sensor_info.header.stamp.secs,
                            sensor_info.header.stamp.nsecs,
                        ]

                    data = pickle.dumps(result)
                    size = sys.getsizeof(data)
                    header = struct.pack("i", size)
                    self.request.sendall(header)
                    self.request.sendall(data)

                elif msg.startswith("[SET_EEF_POSE]"):
                    pose_list = msg.split("[")[-1].split("]")[0]
                    pose_list = pose_list.split(",")
                    pose_list = np.array([float(p) for p in pose_list])

                    if self.mode == POSE:
                        self.robot.action(pose_list)
                    elif self.mode == TRAJECTORY:
                        target_pose_array = np.zeros((DEFAULT_ACT_LEN, 7))
                        for i in range(DEFAULT_ACT_LEN):
                            target_pose_array[i][0] = curr_trans[0]
                            target_pose_array[i][1] = curr_trans[1]
                            target_pose_array[i][2] = curr_trans[2]
                            target_pose_array[i][3] = curr_quat[0]
                            target_pose_array[i][4] = curr_quat[1]
                            target_pose_array[i][5] = curr_quat[2]
                            target_pose_array[i][6] = curr_quat[3]

                        self.robot.action_trajectory(target_pose_array)

                    sensor_info = rospy.wait_for_message("/eval_sensors", EvalSensors)

                    self.request.sendall(
                        f"Finish update target pose {pose_list}".encode(FORMAT)
                    )
                elif msg == "[CLOSE_GRIPPER]":
                    robot_status_topic = "/robot_status"
                    self.gripper.close()
                    self.request.sendall(f"Finish close gripper".encode(FORMAT))
                elif msg == "[OPEN_GRIPPER]":
                    self.gripper.open()
                    self.request.sendall(f"Finish open gripper".encode(FORMAT))
                elif msg == "[CLOSE_ACTION]":
                    sensor_info = rospy.wait_for_message("/eval_sensors", EvalSensors)
                    robot_status = self.process_robot_status(sensor_info.robot_status)
                    curr_trans = list(robot_status[10:13])
                    curr_quat = list(robot_status[13:17])

                    if self.mode == POSE:
                        self.robot.action(curr_trans + curr_quat)
                    elif self.mode == TRAJECTORY:
                        target_pose_array = np.zeros((DEFAULT_ACT_LEN, 7))
                        for i in range(DEFAULT_ACT_LEN):
                            target_pose_array[i][0] = curr_trans[0]
                            target_pose_array[i][1] = curr_trans[1]
                            target_pose_array[i][2] = curr_trans[2]
                            target_pose_array[i][3] = curr_quat[0]
                            target_pose_array[i][4] = curr_quat[1]
                            target_pose_array[i][5] = curr_quat[2]
                            target_pose_array[i][6] = curr_quat[3]

                        self.robot.action_trajectory(target_pose_array)

                    self.request.sendall(
                        f"Finish action, stop publishing wbc command ...".encode(FORMAT)
                    )
                elif msg.startswith("[COMPARE_TASK_IMG]"):
                    self.image_compare = ImageCompare()
                    task_id = int(msg.split("]")[-1].lstrip())
                    # self.image_compare.check_scene(task_id=task_id)
                    os.system(f"python3 /home/robot/catkin_ws/src/real_robot_rollout_public/src/headcam_compare.py --task_id {task_id}")
                else:
                    pass
                msg_flag = False
            if traj_flag:
                assert self.mode == TRAJECTORY
                trajectory = self.traj_meta["traj"]
                target_pose_array = np.zeros((len(trajectory), 7))
                for i in range(len(trajectory)):
                    target_pose_array[i][0] = trajectory[i][0]
                    target_pose_array[i][1] = trajectory[i][1]
                    target_pose_array[i][2] = trajectory[i][2]
                    target_pose_array[i][3] = trajectory[i][3]
                    target_pose_array[i][4] = trajectory[i][4]
                    target_pose_array[i][5] = trajectory[i][5]
                    target_pose_array[i][6] = trajectory[i][6]

                if self.traj_meta["home"] == True:
                    self.robot.action_home(target_pose_array)
                else:
                    self.robot.action_trajectory(target_pose_array)

                self.request.sendall(f"Finish traj pose".encode(FORMAT))
                traj_flag = False


ADDR, PORT, HEADER, FORMAT, DISCONNECT_MSG = socket_conf()

if __name__ == "__main__":
    # print(f"Running on {ADDR}...")
    server = socketserver.ThreadingTCPServer(("192.168.1.11", 10243), TaskSocketServer)
    TaskSocketServer.init(TaskSocketServer, mode=POSE)
    server.serve_forever()
