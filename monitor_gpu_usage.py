# monitor the gpu usage of the training process

import os
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import torch

# get the pid of the training process
def get_pid():
    pid = os.popen('ps -ef | grep "python3 train.py" | grep -v grep | awk \'{print $2}\'').read()
    print('pid: ', pid)
    return pid

# get the gpu usage of the training process
def get_gpu_usage(pid, gpu_id=0):
    gpu_usage = os.popen('nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits -i %d | sed -n 2p' % gpu_id).read()
    print('gpu usage: ', gpu_usage)
    return gpu_usage

# monitor the gpu usage of the training process
def monitor_gpu_usage():
    pid = get_pid()
    pid = pid.strip()
    gpu_usage_list = []
    while True:
        gpu_usage = get_gpu_usage(pid)
        gpu_usage_list.append(gpu_usage)
        print('gpu usage: ', gpu_usage)
        time.sleep(1)
        if gpu_usage == 0:
            break
    return gpu_usage_list

# plot the gpu usage of the training process
def plot_gpu_usage(gpu_usage_list):
    x = np.arange(len(gpu_usage_list))
    y = np.array(gpu_usage_list)
    plt.plot(x, y)
    plt.xlabel('time')
    plt.ylabel('gpu usage')
    plt.title('gpu usage of the training process')
    plt.savefig('gpu_usage.png')

if __name__ == '__main__':
    gpu_usage_list = monitor_gpu_usage()
    plot_gpu_usage(gpu_usage_list)
    