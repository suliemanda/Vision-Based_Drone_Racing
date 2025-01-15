class State():
    def __init__(self):
        self.twist=Twist()
        self.accel=Accel()
        self.pose=Pose()
        self.quadrotor_id=0
class Twist():
    def __init__(self):
        self.angular=Vector3()
        self.linear=Vector3()
class Accel():
    def __init__(self):
        self.linear=Vector3()
        self.angular=Vector3()
class Pose():
    def __init__(self):
        self.orientation=Quaternion()
        self.position=Point()
class Vector3():
    def __init__(self):
        self.x=0.0
        self.y=0.0
        self.z=0.0
class Quaternion():
    def __init__(self):
        self.x=0.0
        self.y=0.0
        self.z=0.0
        self.w=1
class Point():
    def __init__(self):
        self.x=0.0
        self.y=0.0
        self.z=0.0
class Imu():
    def __init__(self):
        self.header=Header()
        self.angular_velocity=Vector3()
        self.linear_acceleration=Vector3()
        self.angular_velocity_covariance=[
            0.0,0.0,0.0,
            0.0,0.0,0.0,
            0.0,0.0,0.0
        ]
        self.linear_acceleration_covariance=[
            0.0,0.0,0.0,
            0.0,0.0,0.0,
            0.0,0.0,0.0
        ]
class Header():
    def __init__(self):
        self.stamp=Time()
class Time():
    def __init__(self):
        self.sec=0
        self.nanosec=0
