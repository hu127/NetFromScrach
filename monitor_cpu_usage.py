# aim to monitor the cpu usage of the training process

import os
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt

# get the pid of the training process
def get_pid():
    pid = os.popen('ps -ef | grep "python3 train.py" | grep -v grep | awk \'{print $2}\'').read()
    return pid

# get the cpu usage of the training process
def get_cpu_usage(pid):
    p = psutil.Process(int(pid))
    cpu_usage = p.cpu_percent(interval=1)
    return cpu_usage

# monitor the cpu usage of the training process
def monitor_cpu_usage():
    pid = get_pid()
    print('pid: ', pid)
    pid = pid.strip()
    cpu_usage_list = []
    while True:
        cpu_usage = get_cpu_usage(pid)
        cpu_usage_list.append(cpu_usage)
        print('cpu usage: ', cpu_usage)
        time.sleep(1)
        if cpu_usage == 0:
            break
    return cpu_usage_list

# plot the cpu usage of the training process
def plot_cpu_usage(cpu_usage_list):
    x = np.arange(len(cpu_usage_list))
    y = np.array(cpu_usage_list)
    plt.plot(x, y)
    plt.xlabel('time')
    plt.ylabel('cpu usage')
    plt.title('cpu usage of the training process')
    plt.savefig('cpu_usage.png')

if __name__ == '__main__':
    cpu_usage_list = monitor_cpu_usage()
    plot_cpu_usage(cpu_usage_list)
    