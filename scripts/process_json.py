import json
import os

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def replace_parent_path(path, old_parent, new_parent):
    return path.replace(old_parent, new_parent)

def process_video_paths(json_dir):
    json_files = os.listdir(json_dir)
    old_parent = "/mnt/bn/robotics-real-data/gr2_data/CALVIN/task_ABCD_D/"
    new_parent = "/data/home/hanbo/projects/RobotVLM/datasets/CALVIN/debug/media/"
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        json_data = load_json(json_path)
        for video in json_data['videos']:
            video['video_path'] = replace_parent_path(video['video_path'], old_parent, new_parent)
            assert os.path.exists(video['video_path'])
        with open(json_path, 'w') as file:
            json.dump(json_data, file, indent=4)

if __name__ == "__main__":
    json_dir = "/data/home/hanbo/projects/RobotVLM/datasets/CALVIN/debug/processed_val/"
    process_video_paths(json_dir)