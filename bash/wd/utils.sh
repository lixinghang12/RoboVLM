#!/bin/bash

# 设置随机种子，基于当前时间
RANDOM_SEED=$(date +%s)

# 函数：获取空闲GPU
get_free_gpu() {
    # 获取所有GPU的信息
    local gpus_info=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

    # 遍历GPU信息
    while IFS=, read -r gpu_id memory_used; do
        # 检查显存占用是否小于1GB
        if [ "$memory_used" -lt 1024 ]; then
            echo $gpu_id
            return
        fi
    done <<< "$gpus_info"

    # 如果没有找到空闲的GPU，则退出
    echo "Error: No GPU with less than 1GB memory used found." >&2
    exit 1
}

# 函数：获取可用端口
get_free_port() {
    # 尝试获取一个随机端口，范围从1024到65535
    local port_range_min=1024
    local port_range_max=65535
    local port

    while true; do
        # 生成一个随机端口
        port=$((port_range_min + RANDOM % (port_range_max - port_range_min + 1)))

        # 检查端口是否被占用
        if ! ss -tl | grep -q ":$port "; then
            echo $port
            return
        fi
    done
}
