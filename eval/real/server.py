import argparse
import base64
from io import BytesIO
from eval.real.rollout import Rollout
from flask import Flask, request, jsonify, session
from pytorch_lightning import seed_everything
import os
import numpy as np


app = Flask(__name__)

seed_everything(0)
parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
parser.add_argument('--debug', action="store_true", help="Print debug info and visualize environment.")
parser.add_argument('--config_path', type=str, default=None, help='path to the config file')
parser.add_argument('--ckpt_dir', type=str, nargs='+', default="", help="checkpoint directory of the training")
parser.add_argument('--ckpt_path', type=str, default=None, help="checkpoint directory of the training")
parser.add_argument('--ckpt_idx', type=int, default=-1,help="which ckpt is going to be evaluated")
parser.add_argument('--device_id', default=0, type=int, help="CUDA device")
parser.add_argument('--no_cache', action="store_true")
parser.add_argument('--debug_model', action="store_true")
args = parser.parse_args()

rollout_instance = Rollout(
    config_path=args.config_path,
    ckpt_dir=args.ckpt_dir,
    ckpt_path=args.ckpt_path,
    ckpt_idx=args.ckpt_idx,
    device_id=args.device_id,
    debug_model=args.debug_model,
    no_cache=args.no_cache,
)

def load(_data):
    with BytesIO(base64.urlsafe_b64decode(_data)) as f:
        data = np.load(f, allow_pickle=True)
    return data

@app.route('/', methods=['GET'])
def step_api():
    data = request.get_json()
    if 'robot_state' not in data or 'static_rgb' not in data or 'hand_rgb' not in data or 'text' not in data:
        return jsonify({'error': 'Invalid data'}), 400
    for key in ['robot_state', 'static_rgb', 'hand_rgb']:
        data[key] = load(data[key])
    action: np.ndarray = rollout_instance.step(data)
    return jsonify(action.tolist())    

if __name__ == '__main__':
    port = int(os.environ['ARNOLD_WORKER_0_PORT'])
    app.run(host="::", port=port, debug=False)
