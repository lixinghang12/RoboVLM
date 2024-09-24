import subprocess
import time
import os
from pathlib import Path
import torch

def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    return memory_used_values

def main():
    gpu_num = torch.cuda.device_count()
    max_memory = [0]*gpu_num  # Assuming you have 8 GPUs
    dir_name = Path('gpu_detect')
    os.makedirs(dir_name, exist_ok=True)
    while True:
        memory_used = get_gpu_memory()
        for i in range(gpu_num):  # Again assuming you have 8 GPUs
            if memory_used[i] > max_memory[i]:
                max_memory[i] = memory_used[i]
                # with open(f'log{i}.txt', 'a') as f:
                #     f.write(f"Max memory used on GPU {i}: {max_memory[i]} MiB\n")
                with open(dir_name/ f'log{i}.txt', 'w') as file:
                    file.write(f"Max memory used on GPU {i}: {max_memory[i]} MiB\n")
        time.sleep(1)

if __name__ == "__main__":
    main()
