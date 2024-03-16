#!/home/wangchuang/anaconda3/envs/pytorchgpupy3.7/bin/python

import rospy

import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

#import seaborn as sns; sns.set()
def path_reading(dir):
    data = pd.read_csv(dir)
    print(np.array(data))
    '''
    newfile = open('/home/wang/catkin_ws/src/wxm_assembly/wxm_robot_teleop/scripts/path_point.csv','r')
    filereader = csv.reader(newfile)
    for rows in filereader:
        print(rows)
    '''
    return np.array(data)


def data_plot(data):
    fig = plt.figure(num=1, dpi=300)
    #sns.set_style("white")
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    #ax4 = fig.add_subplot(4, 2, 4)

    #plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    ax1.cla()
    ax2.cla()
    ax3.cla()
    #ax4.cla()

    t = range(0, len(data))

    ax1.plot(t, data[:, 1], label='tool_x')
    ax1.plot(t, data[:, 2], label='tool_y')
    ax1.plot(t, data[:, 3], label='tool_z')
    #ax1.set_title('position')
    ax1.set_xlabel('s')
    ax1.set_ylabel('Position(m)')
    ax1.legend(ncol=3,loc="upper right")
    ax1.set_ylim(-0.5, 0.5)

    v, a, via_index = via_point(data)
    print(np.size(v))
    t = range(0, np.size(v))
    ax2.plot(t, v, label='velocity')
    #ax2.set_title('velocity')
    ax2.set_xlabel('s')
    ax2.set_ylabel('Velocity')
    #ax2.legend()
    print(np.size(a))
    t = range(0, np.size(a))
    ax3.plot(t, a, label='velocity')
    # ax2.set_title('velocity')
    ax3.set_xlabel('s')
    ax3.set_ylabel('Acceleration')
    #ax3.legend()
    '''ax2.plot(t, data[:, 19], label='speed_x')
    ax2.plot(t, data[:, 20], label='speed_y')
    ax2.plot(t, data[:, 21], label='speed_z')
    ax2.set_title('velocity')
    ax2.set_xlabel('time')
    ax2.set_ylabel('meter')
    ax2.legend()
    ax3.plot(t, data[:, 13], label='wrench_x')
    ax3.plot(t, data[:, 14], label='wrench_y')
    ax3.plot(t, data[:, 15], label='wrench_z')
    ax3.set_title('wrench')
    ax3.set_xlabel('time')
    ax3.set_ylabel('force')
    ax3.legend()'''
    plt.savefig('visualize_demonstration.pdf', dpi=300)
    plt.show()

def moving_average(raw_v, w):
    return np.convolve(raw_v, np.ones(w), "valid") / w


def exponential_moving_average(raw_v):
    v_ema = []
    v_pre = 0
    beta = 0.9
    for i, t in enumerate(raw_v):
        v_t = beta * v_pre + (1 - beta) * t
        v_ema.append(v_t)
        v_pre = v_t
    #print("v_mea:", v_ema)

    v_ema_corr = []
    for i, t in enumerate(v_ema):
        v_ema_corr.append(t / (1 - np.power(beta, i+1)))

    return v_ema_corr

def differential(data):
    raw_v = []
    for i in range(0, len(data)-1):
        tem_v = np.linalg.norm(data[i+1] - data[i])
        raw_v.append(tem_v)
    return(raw_v)

def via_point(data):
    raw_v = differential(data[:, 1:4])

    v_avg = moving_average(raw_v, 4)

    raw_a = differential(v_avg)

    via_index = np.argmax(raw_a)
    print(via_index, np.linalg.norm(data[via_index, 1:4]-data[-1, 1:4]), [data[0, 1:7], data[via_index, 1:7], data[-1, 1:7]])

    return v_avg, raw_a, via_index


def main():
    try:
        data = path_reading("./path_point_for_ILRRL8-peg.csv")
        #via_point(data)
        data_plot(data)
    except KeyboardInterrupt:
        
        rospy.signal_shutdown("KeyboardInterrupt")
        raise


if __name__ == '__main__': main()