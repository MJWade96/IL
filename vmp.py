#!/home/wangchuang/anaconda3/envs/pytorchgpupy3.7/bin/python
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

pinv_rcond = 1.4e-08


class VMP:
    def __init__(self, dim, kernel_num=30, kernel_std=0.1, elementary_type='linear', use_out_of_range_kernel=True):
        self.kernel_num = kernel_num
        if use_out_of_range_kernel:
            self.centers = np.linspace(1.2, -0.2, kernel_num)  # (K, )
        else:
            self.centers = np.linspace(1, 0, kernel_num)  # (K, )

        self.kernel_variance = kernel_std ** 2
        self.var_reci = - 0.5 / self.kernel_variance
        self.elementary_type = elementary_type
        self.lamb = 0.01
        self.dim = dim
        self.n_samples = 100
        self.kernel_weights = np.zeros(shape=(kernel_num, self.dim))

        self.h_params =None
        self.y0 = None
        self.g = None

    def __psi__(self, can_value: Union[float, np.ndarray]):
        """
        compute the contribution of each kernel given a canonical value
        """
        #ic(can_value - self.centers)
        return np.exp(np.square(can_value - self.centers) * self.var_reci)

    def __Psi__(self, can_values: np.ndarray):
        """
        compute the contributions of each kernel at each time step as a (T, K) matrix, where
        can_value is a (T, ) array, the sampled canonical values, where T is the total number of time steps.
        """
        return self.__psi__(can_values[:, None])

    def h(self, x):
        if self.elementary_type == 'linear':
            return np.matmul(self.h_params, np.matrix([[1], [x]]))
        else:
            return np.matmul(self.h_params, np.matrix(
                [[1], [x], [np.power(x, 2)], [np.power(x, 3)], [np.power(x, 4)], [np.power(x, 5)]]))

    def linear_traj(self, can_values: np.ndarray):
        """
        compute the linear trajectory (T, dim) given canonical values (T, )
        """
        if self.elementary_type == 'linear':
            can_values_aug = np.stack([np.ones(can_values.shape[0]), can_values])
        else:
            can_values_aug = np.stack([np.ones(can_values.shape[0]), can_values, np.power(can_values, 2),
                                       np.power(can_values, 3), np.power(can_values, 4), np.power(can_values, 5)])
        #ic(self.h_params, can_values_aug)
        return np.einsum("ij,ik->kj", self.h_params, can_values_aug)  # (n, 2) (T, 2)

    def train(self, trajectories):
        """
        Assume trajectories are regularly sampled time-sequences.
        """
        if len(trajectories.shape) == 2: # (n, T, 2)
            trajectories = np.expand_dims(trajectories, 0)

        n_demo, self.n_samples, self.dim = trajectories.shape
        self.dim -= 1

        can_value_array = self.can_sys(1, 0, self.n_samples)  # canonical variable (T)
        Psi = self.__Psi__(can_value_array)  # (T, K) squared exponential (SE) kernels

        if self.elementary_type == 'linear':
            y0 = trajectories[:, 0, 1:].mean(axis=0)
            g = trajectories[:, -1, 1:].mean(axis=0)
            self.h_params = np.stack([g, y0-g])

        else:
            # min_jerk
            y0 = trajectories[:, 0:3, 1:].mean(axis=0)
            g = trajectories[:, -2:, 1:].mean(axis=0)
            dy0 = (y0[1, 2:] - y0[0, 2:]) / (y0[1, 1] - y0[0, 1])
            dy1 = (y0[2, 2:] - y0[1, 2:]) / (y0[2, 1] - y0[1, 1])
            ddy0 = (dy1 - dy0) / (y0[1, 1] - y0[0, 1])
            dg0 = (g[1, 2:] - g[0, 2:]) / (g[1, 1] - g[0, 1])
            dg1 = (g[2, 2:] - g[1, 2:]) / (g[2, 1] - g[1, 1])
            ddg = (dg1 - dg0) / (g[1, 1] - g[0, 1])

            b = np.stack([y0[0, :], dy0, ddy0, g[-1, :], dg1, ddg])
            A = np.array([[1, 1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5], [0, 0, 2, 6, 12, 20], [1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0]])
            self.h_params = np.linalg.solve(A, b)

        self.y0 = y0
        self.g = g
        linear_traj = self.linear_traj(can_value_array)  # (T, dim)  elementary trajectory
        shape_traj = trajectories[..., 1:] - np.expand_dims(linear_traj, 0)  # (N, T, dim) - (1, T, dim) shape modulation

        pseudo_inv = np.linalg.pinv(Psi.T.dot(Psi), pinv_rcond)  # (K, K)
        self.kernel_weights = np.einsum("ij,njd->nid", pseudo_inv.dot(Psi.T), shape_traj).mean(axis=0)  #
        ic(Psi.shape, shape_traj.shape, self.kernel_weights.shape)
        return linear_traj

    def save_weights_to_file(self, filename):
        np.savetxt(filename, self.kernel_weights, delimiter=',')

    def load_weights_from_file(self, filename):
        self.kernel_weights = np.loadtxt(filename, delimiter=',')

    def get_weights(self):
        return self.kernel_weights

    def get_flatten_weights(self):
        return self.kernel_weights.flatten('F')

    def set_weights(self, ws: np.ndarray):
        """
        set weights to VMP

        Args:
            ws: (kernel_num, dim)
        """
        if np.shape(ws)[-1] == self.dim * self.kernel_num:
            self.kernel_weights = np.reshape(ws, (self.kernel_num, self.dim), 'F')
        elif np.shape(ws)[0] == self.kernel_num and np.shape(ws)[-1] == self.dim:
            self.kernel_weights = ws
        else:
            raise Exception(f"The weights have wrong shape. "
                            f"It should have {self.kernel_num} rows (for kernel number) "
                            f"and {self.dim} columns (for dimensions), but given is {ws.shape}.")

    def get_position(self, t):
        x = 1 - t
        return np.matmul(self.__psi__(x), self.kernel_weights)

    def set_start(self, y0):
        self.y0 = y0
        self.h_params = np.stack([self.g, self.y0 - self.g])

    def set_goal(self, g):
        self.g = g
        self.h_params = np.stack([self.g, self.y0 - self.g])

    def set_start_goal(self, y0, g):
        self.y0 = y0
        self.g = g
        self.h_params = np.stack([self.g, self.y0 - self.g])

    # def set_start_goal(self, y0, g, dy0=None, dg=None, ddy0=None, ddg=None):
    #     self.y0 = y0
    #     self.g = g
    #     self.q0 = y0
    #     self.q1 = g
    #
    #     self.goal = g
    #     self.start = y0
    #
    #
    #     if self.ElementaryType == "minjerk":
    #         zerovec = np.zeros(shape=np.shape(self.y0))
    #         if dy0 is not None and np.shape(dy0)[0] == np.shape(self.y0)[0]:
    #             dy0 = dy0
    #         else:
    #             dy0 = zerovec
    #
    #         if ddy0 is not None and np.shape(ddy0)[0] == np.shape(self.y0)[0]:
    #             ddy0 = ddy0
    #         else:
    #             ddy0 = zerovec
    #
    #         if dg is not None and np.shape(dg)[0] == np.shape(self.y0)[0]:
    #             dg = dg
    #         else:
    #             dg = zerovec
    #
    #         if ddg is not None and np.shape(ddg)[0] == np.shape(self.y0)[0]:
    #             ddg = ddg
    #         else:
    #             ddg = zerovec
    #
    #         self.h_params = self.get_min_jerk_params(self.y0 , self.g, dy0=dy0, dg=dg, ddy0=ddy0, ddg=ddg)
    #     else:
    #         self.h_params = np.transpose(np.stack([self.g, self.y0 - self.g]))

    def roll(self, y0, g, n_samples=None):
        """
        reproduce the trajectory given start point y0 (dim, ) and end point g (dim, ), return traj (n_samples, dim)
        """
        n_samples = self.n_samples if n_samples is None else n_samples
        can_values = self.can_sys(1, 0, n_samples)  # canonical variable (T)

        if self.elementary_type == "minjerk":
            dv = np.zeros(y0.shape)
            self.h_params = self.get_min_jerk_params(y0, g, dv, dv, dv, dv)
        else:
            self.h_params = np.stack([g, y0 - g])

        linear_traj = self.linear_traj(can_values) # (T, dim)  elementary trajectory

        psi = self.__Psi__(can_values)  # (T, K) squared exponential (SE) kernels
        ic(psi.shape, self.kernel_weights.shape)
        traj = linear_traj + np.einsum("ij,jk->ik", psi, self.kernel_weights)

        time_stamp = 1 - np.expand_dims(can_values, 1)
        return np.concatenate([time_stamp, traj], axis=1), linear_traj

    def get_target(self, t):
        action = np.transpose(self.h(1-t)) + self.get_position(t)
        return action

    @staticmethod
    def can_sys(t0, t1, n_sample):
        """
        return the sampled values of linear decay canonical system

        Args:
            t0: start time point
            t1: end time point
            n_sample: number of samples
        """
        return np.linspace(t0, t1, n_sample)

    @staticmethod
    def get_min_jerk_params(y0, g, dy0, dg, ddy0, ddg):
        b = np.stack([y0, dy0, ddy0, g, dg, ddg])
        A = np.array(
            [[1, 1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5], [0, 0, 2, 6, 12, 20], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
             [0, 0, 2, 0, 0, 0]])

        return np.linalg.solve(A, b)


# define the transform matrix
def transformation_matrix(position, euler_angles):
    # position transform matrix
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = position

    # angle transform matrix
    roll, pitch, yaw = euler_angles
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = np.array([
        [np.cos(yaw)*np.cos(pitch), -np.sin(yaw)*np.cos(roll)+np.cos(yaw)*np.sin(pitch)*np.sin(roll), np.sin(yaw)*np.sin(roll)+np.cos(yaw)*np.sin(pitch)*np.cos(roll)],
        [np.sin(yaw)*np.cos(pitch), np.cos(yaw)*np.cos(roll)+np.sin(yaw)*np.sin(pitch)*np.sin(roll), -np.cos(yaw)*np.sin(roll)+np.sin(yaw)*np.sin(pitch)*np.cos(roll)],
        [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]
    ])

    # pose transform matrix
    transformation_matrix = np.dot(translation_matrix, rotation_matrix)
    return transformation_matrix



if __name__ == '__main__':
    


    ################################ test with real data ############################
    #from robot_utils.py.visualize import set_3d_equal_auto
    traj_files = ["./path_point_for_ILRRL1.csv"]


    trajs = np.array([np.loadtxt(f, delimiter=',') for f in traj_files])
    ic(trajs.shape)

    trajs2 = trajs[:, :98, :]
    ic(trajs2.shape)

    # preprocess for task-centric
    #trajs2 = np.concatenate([np.expand_dims(trajs[:, :90, 0], 2), trajs[:, :90, 1:4] - trajs[:, :90, 7:10]], axis=2)
    #ic(trajs2[:, 0, 1:7])
    '''
    transformation_matrix_robot_to_object = np.linalg.inv(transformation_matrix(trajs[0, 0, 7:10], trajs[0, 0, 10:13])) #position_object, euler_angles_object
    transformation_matrix_ee_to_robot = transformation_matrix(trajs[0, 0, 1:4], trajs[0, 0, 4:7])
    pose_object = np.dot(transformation_matrix_robot_to_object, transformation_matrix_ee_to_robot)
    ic(pose_object) #pose_object[:, 0, 1:7]
    '''


    #########via-point#########
    # via-point extraction and task-centric
    via_t = [0, 23, trajs2[0, :, 1].shape[0]-1]
    off_set = [trajs[:, via_t[1], 1:4] - trajs[:, via_t[2], 7:10], trajs[:, via_t[2], 1:4] - trajs[:, via_t[2], 7:10]]

    # training
    vmp_set = []
    linear_traj_raw = trajs[:, 0, 1:4]
    for i in range(len(via_t)-1):
        vmp = VMP(3, kernel_num=int(0.5*via_t[i+1]), elementary_type='linear', use_out_of_range_kernel=False)
        temp_linear_traj_raw = vmp.train(trajs2[:, via_t[i]:via_t[i+1], 0:4])
        vmp_set.append(vmp)
        linear_traj_raw = np.concatenate((linear_traj_raw, temp_linear_traj_raw), axis=0)

    # scale to variable position [0.03, 0.03, 0]
    # via_point modulation
    start = trajs[:, 0, 1:4][0]
    task = trajs[:, -1, 7:10][0] + np.array([0.03, 0.03, 0])
    via_point = [start, off_set[0][0] + task, off_set[1][0] + task]
    #via_point = [start, trajs2[0, via_point[1], 1:4] + task, trajs2[0, via_point[2], 1:4] + task]
    ic(via_point)

    # reproduce
    scaled_VMP_p003 = trajs[:, 0, 0:4]
    linear_traj = trajs[:, 0, 1:4]
    for i in range(len(via_point)-1):
        temp_reproduced, temp_linear_traj = vmp_set[i].roll(via_point[i], via_point[i+1], via_t[i+1]-via_t[i])
        ic(temp_reproduced, temp_linear_traj)
        # planned trajectory is directly used as the base trajectory in transfer phase with index col for alignment
        if i < 1:
            temp_reproduced = np.insert(temp_linear_traj, 0, np.linspace(0, 1, temp_linear_traj.shape[0]), axis=1)
        scaled_VMP_p003 = np.concatenate((scaled_VMP_p003, temp_reproduced), axis=0)
        linear_traj = np.concatenate((linear_traj, temp_linear_traj), axis=0)

    # scale to variable position [-0.03, -0.03, 0]
    # via_point modulation
    start = trajs[:, 0, 1:4][0]
    task = trajs[:, -1, 7:10][0] + np.array([-0.03, -0.03, 0])
    via_point = [start, off_set[0][0] + task, off_set[1][0] + task]
    # via_point = [start, trajs2[0, via_point[1], 1:4] + task, trajs2[0, via_point[2], 1:4] + task]
    ic(via_point)

    # reproduce
    scaled_VMP_n003 = trajs[:, 0, 0:4]
    linear_traj = trajs[:, 0, 1:4]
    for i in range(len(via_point) - 1):
        temp_reproduced, temp_linear_traj = vmp_set[i].roll(via_point[i], via_point[i + 1], via_t[i+1]-via_t[i])
        # planned trajectory is directly used as the base trajectory in transfer phase with index col for alignment
        if i < 1:
            temp_reproduced = np.insert(temp_linear_traj, 0, np.linspace(0, 1, temp_linear_traj.shape[0]), axis=1)
        scaled_VMP_n003 = np.concatenate((scaled_VMP_n003, temp_reproduced), axis=0)
        linear_traj = np.concatenate((linear_traj, temp_linear_traj), axis=0)

    ic(reproduced.shape)
    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.6)

    ax1.plot(trajs[0, :, 1], trajs[0, :, 2], trajs[0, :, 3], color='blue', label='Demonstration')
    ax1.plot(scaled_VMP_p003[:, 1], scaled_VMP_p003[:, 2], scaled_VMP_p003[:, 3], color="red", label='VMP')
    ax1.plot(scaled_VMP_n003[:, 1], scaled_VMP_n003[:, 2], scaled_VMP_n003[:, 3], color="red")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    #set_3d_equal_auto(ax)
    ax1.legend(ncol=2, loc="upper right")
    #ic(trajs[0, :, :], reproduced[:, :])


    t = np.linspace(0, 1, trajs2[0, :, 1].shape[0])

    ax2.plot(t, trajs2[0, :, 1], color="b", linestyle="-", label='Demonstration')
    #ax2.plot(t, linear_traj_raw[:, 0], color="b", linestyle="-.")

    ax2.plot(t, linear_traj[:, 0], color="r", linestyle="-.", label='VMP_h(x)')
    ax2.plot(t, scaled_VMP_n003[:, 1], color="r", linestyle="-", alpha=0.5, label='VMP')

    ax2.plot(t, linear_traj_DMP[:, 0], color="g", linestyle="-.", label='DMP_h(x)')
    ax2.plot(t, scaled_DMP_n003[:, 1], color="g", linestyle="-", alpha=0.5, label='DMP')
    #ax2.set_xlabel("t")
    #ax2.set_ylabel("x")
    ax2.legend(ncol=1, loc="upper right")
    '''
    plt.plot(t, trajs2[0, :, 1], color="r", linestyle="--")
    #plt.plot(t, trajs2[0, :, 2], color="g", linestyle="--")
    #plt.plot(t, trajs2[0, :, 3], color="b", linestyle="--")
    
    #plt.plot(t, linear_traj_raw[:, 0], color="r", linestyle="-.")
    #plt.plot(t, linear_traj_raw[:, 1], color="g", linestyle="-.")
    #plt.plot(t, linear_traj_raw[:, 2], color="b", linestyle="-.")

    plt.plot(t, linear_traj[:, 0], color="r", linestyle="-.")
    #plt.plot(t, linear_traj[:, 1], color="g", linestyle="-.")
    #plt.plot(t, linear_traj[:, 2], color="b", linestyle="-.")

    plt.plot(t, scaled_VMP_n003[:, 1], color="r", linestyle="-", alpha=0.5)
    #plt.plot(t, scaled_VMP_n003[:, 2], color="g", linestyle="-", alpha=0.5)
    #plt.plot(t, scaled_VMP_n003[:, 3], color="b", linestyle="-", alpha=0.5)

    plt.plot(t, scaled_DMP_n003[:, 1], color="r", linestyle="-", alpha=0.5)
    #plt.plot(t, scaled_DMP_n003[:, 2], color="g", linestyle="-", alpha=0.5)
    #plt.plot(t, scaled_DMP_n003[:, 3], color="b", linestyle="-", alpha=0.5)
    
    ax2 = fig.add_subplot(133)
    plt.plot(t, trajs2[0, :, 3] - linear_traj_raw[:, 2], color="r", linestyle="-.")
    plt.plot(reproduced[:, 0], reproduced[:, 3] - linear_traj[:, 2], color="r", linestyle="-", alpha=0.5)
    '''
    plt.savefig('visualize_IL_real_data.png', dpi=300)
    plt.show()



