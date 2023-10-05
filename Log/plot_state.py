import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


fastlio_log = pd.read_csv('./fast_lio_log.csv', delimiter=',')
timestamp = fastlio_log['timestamp']
preprocess_time = fastlio_log['preprocess time']
incremental_time = fastlio_log['incremental time']
search_time = fastlio_log['search time']
total_time = fastlio_log['total time']

# keyframe_pose = np.loadtxt('./keyframe_pose.txt', dtype=np.float32, delimiter=' ')
keyframe_pose_optimized = np.loadtxt('./keyframe_pose_optimized.txt', dtype=np.float32, delimiter=' ')
# gnss_pose = np.loadtxt('./gnss_pose.txt', dtype=np.float32, delimiter=' ')
state_predict = np.loadtxt('./state_predict.txt', dtype=np.float32, delimiter=' ')
state_update = np.loadtxt('./state_update.txt', dtype=np.float32, delimiter=' ')
imu_quat_eular = np.loadtxt('./imu_quat_eular.txt', dtype=np.float32, delimiter=' ')


def fast_lio_time_log_analysis():
    plt.plot(timestamp, preprocess_time, label='preprocess_time')
    plt.plot(timestamp, incremental_time, label='incremental_time')
    plt.plot(timestamp, search_time, label='search_time')
    plt.plot(timestamp, total_time, label='total_time')

    # plt.xlim(0, 1000)
    # plt.ylim(0, 200)

    plt.xlabel('timestamp')
    plt.ylabel('cost times(ms)')
    plt.legend()
    plt.title('fast_lio_time_log_analysis')
    plt.show()


def system_state_analysis():
    fig, axs = plt.subplots(4, 2)
    lab_pre = ['', 'pre-x', 'pre-y', 'pre-z']
    lab_out = ['', 'out-x', 'out-y', 'out-z']
    time = state_predict[:, 0]
    axs[0, 0].set_title('imu_rpy')
    axs[1, 0].set_title('imu_xyz')
    axs[2, 0].set_title('imu_velocity')
    axs[3, 0].set_title('bg')
    axs[0, 1].set_title('ba')
    axs[1, 1].set_title('ext_R')
    axs[2, 1].set_title('ext_T')
    axs[3, 1].set_title('gravity')
    index = [0, 1, 2, 5, 6, 7, 8, 9]
    for i in range(1, 4):
        for j in range(8):
            axs[j % 4, j // 4].plot(time, state_predict[:, i + index[j] * 3], label=lab_pre[i])
            axs[j % 4, j // 4].plot(time, state_update[:, i + index[j] * 3], label=lab_out[i])
            # axs[j % 4, j // 4].plot(time, state_predict[:, i + j * 3], '.-', label=lab_pre[i])
    for j in range(8):
        axs[j % 4, j // 4].grid()
        axs[j % 4, j // 4].legend()
    plt.suptitle("system_state_analysis")
    plt.show()


def z_axis_drift_raw_state_analysis():
    fig, axs = plt.subplots(2, 2)
    lab_out = ['', 'out-x', 'out-y', 'out-z']
    time = state_update[:, 0]
    time2 = imu_quat_eular[:, 0]
    axs[0, 0].set_title('imu_position')
    axs[1, 0].set_title('ba')
    # axs[2, 0].set_title('imu_velocity')
    axs[0, 1].set_title('imu_eular')
    axs[1, 1].set_title('bg')
    axs[0, 0].plot(time, state_update[:, 6], label='z-axis')
    axs[0, 1].plot(time, state_update[:, 1], label='roll')     # x-axis
    axs[0, 1].plot(time, state_update[:, 2], label='pitch')    # y-axis
    # axs[0, 1].plot(time2, imu_quat_eular[:, 4], label='roll-filter')
    axs[0, 1].plot(time2, imu_quat_eular[:, 5], label='pitch-filter')

    axs[1, 1].plot(time, state_update[:, 16], label=lab_out[1])
    axs[1, 1].plot(time, state_update[:, 17], label=lab_out[2])
    axs[1, 1].plot(time, state_update[:, 18], label=lab_out[3])

    axs[1, 0].plot(time, state_update[:, 19], label=lab_out[1])
    axs[1, 0].plot(time, state_update[:, 20], label=lab_out[2])
    axs[1, 0].plot(time, state_update[:, 21], label=lab_out[3])

    for j in range(4):
        axs[j % 2, j // 2].grid()
        axs[j % 2, j // 2].legend()
    plt.suptitle("z_axis_drift_raw_state_analysis")
    plt.show()


def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    return euler


def z_axis_drift_optimized_state_analysis():
    fig, axs = plt.subplots(1, 2)
    axs[0].set_title('imu_position')
    axs[1].set_title('imu_eular')
    timestamps = np.arange(keyframe_pose_optimized[:, 0].shape[0])

    quat_x = keyframe_pose_optimized[:, 4]
    quat_y = keyframe_pose_optimized[:, 5]
    quat_z = keyframe_pose_optimized[:, 6]
    quat_w = keyframe_pose_optimized[:, 7]

    roll = []
    pitch = []
    yaw = []
    for index, value in enumerate(quat_x):
        euler = quaternion2euler([quat_x[index], quat_y[index], quat_z[index], quat_w[index]])
        roll.append(euler[0])
        pitch.append(euler[1])
        yaw.append(euler[2])
    axs[0].plot(timestamps, keyframe_pose_optimized[:, 3], label='z-axis')
    axs[1].plot(timestamps, roll, label='roll')     # x-axis
    axs[1].plot(timestamps, pitch, label='pitch')    # y-axis

    axs[0].grid()
    axs[1].legend()
    plt.suptitle("z_axis_drift_optimized_state_analysis")
    plt.show()


# fast_lio_time_log_analysis()
# system_state_analysis()
# z_axis_drift_raw_state_analysis()
z_axis_drift_optimized_state_analysis()
