# monitor the memory usage of the training process

import os
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt

# get the pid of the training process
def get_pid():
    pid = os.popen('ps -ef | grep "python3 train.py" | grep -v grep | awk \'{print $2}\'').read()
    return pid

# get the memory usage of the training process
def get_memory_usage(pid):
    # print('pid: ', pid)
    p = psutil.Process(int(pid))
    memory_usage = p.memory_info().rss / 1024 / 1024
    return memory_usage

# monitor the memory usage of the training process
def monitor_memory_usage():
    pid = get_pid()
    pid = pid.strip()
    memory_usage_list = []
    while True:
        memory_usage = get_memory_usage(pid)
        memory_usage_list.append(memory_usage)
        print('memory usage: ', memory_usage, 'MB')
        time.sleep(1)
        if memory_usage == 0:
            break
    return memory_usage_list

# plot the memory usage of the training process
def plot_memory_usage(memory_usage_list):
    x = np.arange(len(memory_usage_list))
    y = np.array(memory_usage_list)
    plt.plot(x, y)
    plt.xlabel('time')
    plt.ylabel('memory usage, MB')
    plt.title('memory usage of the training process')
    plt.savefig('memory_usage.png')

if __name__ == '__main__':
    memory_usage_list = monitor_memory_usage()
    plot_memory_usage(memory_usage_list)