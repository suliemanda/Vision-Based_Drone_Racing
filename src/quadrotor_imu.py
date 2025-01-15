import numpy as np
from scipy.spatial.transform import Rotation


try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()
from .types import State, Imu

DEFAULT_FREQUENCY = 100.0  # Hz





class QuadrotorIMU():
    def __init__(self, suffix='', **kwargs):
        self.imu_publishing_frequency =kwargs.get('imu_publishing_frequency', DEFAULT_FREQUENCY)
        self.include_gravity = kwargs.get('include_gravity', True)
        self.angular_velocity_mean = kwargs.get('angular_velocity_mean', [0.001, 0.001, 0.01])
        self.linear_acceleration_mean = kwargs.get('linear_acceleration_mean', [0.2, 0.2, 0.1])
        self.angular_velocity_covariance = kwargs.get('angular_velocity_covariance', [1.7e-4, 0.0, 0.0, 0.0, 1.7e-4, 0.0, 0.0, 0.0, 1.7e-4])
        self.linear_acceleration_covariance = kwargs.get('linear_acceleration_covariance', [2.0000e-3, 0.0, 0.0, 0.0, 2.0000e-3, 0.0, 0.0, 0.0, 2.0000e-3])
        self.angular_velocity_random_walk = kwargs.get('angular_velocity_random_walk', [1.9393e-05, 0.0, 0.0, 0.0, 1.9393e-05, 0.0, 0.0, 0.0, 1.9393e-05])
        self.linear_acceleration_random_walk = kwargs.get('linear_acceleration_random_walk', [3.0000e-03, 0.0, 0.0, 0.0, 3.0000e-03, 0.0, 0.0, 0.0, 3.0000e-03])

        self.initialize_data()


    def initialize_data(self):
        self.state = State()
        self.imu = Imu()
        self.imu.angular_velocity_covariance = np.array(self.angular_velocity_covariance, dtype=np.float64)
        self.imu.linear_acceleration_covariance = np.array(self.linear_acceleration_covariance, dtype=np.float64)

    def receive_state_callback(self, msg: State):
        self.state = msg
        self.publish_imu_callback()

    def publish_imu_callback(self):
        w_act = np.array([self.state.twist.angular.x, self.state.twist.angular.y, self.state.twist.angular.z])
        GyroWalk=np.array(self.angular_velocity_random_walk).reshape((3, 3))
        w_drift = np.array(self.angular_velocity_mean)+np.random.multivariate_normal(np.array([0.0,0.0,0.0]),GyroWalk)
        w_noise_cov = np.array(self.angular_velocity_covariance).reshape((3, 3))
        w_imu = (np.random.multivariate_normal(w_act, w_noise_cov)+w_drift).flatten()

        a_W_act = np.array([self.state.accel.linear.x, self.state.accel.linear.y, self.state.accel.linear.z])
        if self.include_gravity:
            a_W_act[2] += 9.81  # Add gravity
        q_act = np.array([self.state.pose.orientation.x, self.state.pose.orientation.y,
                         self.state.pose.orientation.z, self.state.pose.orientation.w])
        rot = Rotation.from_quat(q_act)
        a_B_act = rot.inv().apply(a_W_act)
        AccWalk=np.array(self.linear_acceleration_random_walk).reshape((3, 3))
        a_drift = np.array(self.linear_acceleration_mean)+np.random.multivariate_normal(np.array([0.0,0.0,0.0]),AccWalk)
        a_noise_cov = np.array(self.linear_acceleration_covariance).reshape((3, 3))
        a_imu = (np.random.multivariate_normal(a_B_act, a_noise_cov)+a_drift).flatten()

        self.imu.angular_velocity.x = w_imu[0]
        self.imu.angular_velocity.y = w_imu[1]
        self.imu.angular_velocity.z = w_imu[2]
        self.imu.linear_acceleration.x = a_imu[0]
        self.imu.linear_acceleration.y = a_imu[1]
        self.imu.linear_acceleration.z = a_imu[2]
