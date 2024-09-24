import base64
from io import BytesIO
from flask import Flask, request, jsonify
import numpy as np
import os

app = Flask(__name__)

def load(_data):
    with BytesIO(base64.urlsafe_b64decode(_data)) as f:
        data = np.load(f, allow_pickle=True)
    return data

@app.route('/', methods=['GET'])
def step_api():
    data = request.get_json()
    if 'robot_state' not in data or 'static_rgb' not in data or 'hand_rgb' not in data or 'text' not in data:
        return jsonify({'error': 'Invalid data'}), 400
    robot_state = load(data['robot_state'])
    static_rgb = load(data['static_rgb'])
    hand_rgb = load(data['hand_rgb'])
    text = data['text']
    import pdb; pdb.set_trace()
    action = np.array([1,2,3])
    return jsonify(action)

if __name__ == '__main__':
    port = int(os.environ['ARNOLD_WORKER_0_PORT'])
    app.run(host="::", port=port, debug=False)
