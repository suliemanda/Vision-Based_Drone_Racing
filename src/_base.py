from geometry_msgs.msg import Wrench
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .types import  State
from scipy.spatial.transform import Rotation
import yaml
import pyquaternion
from .quadrotor_imu import QuadrotorIMU
from .quadrotor_pybullet_camera import QuadrotorPybulletCamera
from .quadrotor_pybullet_physics import QuadrotorPybulletPhysics
from .track import Track


class QuadrotorBaseEnv(gym.Env):

    num_envs = 0

    def __init__(self, sensors=[], env_suffix='', config=None, time_limit=15, terminate_on_contact=False,):
        if isinstance(config, dict):
            config = config
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        

        parameters_physics = {
            'render_ground': False,
            'simulation_step_frequency': 100,
            'obstacles_description': ['gap', 'gap', 'gap', 'gap', 'gap', 'gap', 'gap'],
            'obstacles_poses': [
            -0.6, -0.86, 4.68, 0.0, 0.0, -0.1736482, 0.9848078,
            9.0, 6.45, 2.05, 0.0, 0.0, 0.0, 1.0,
            8.85, -3.8, 2.05, 0.0, 0.0, -0.9063078, 0.4226183,
            -4.3, -5.6, 4.4, 0.0, 0.0, -1.0, 0.0,
            -4.3, -5.6, 2.42, 0.0, 0.0, 0.0, 1.0,
            4.5, -0.45, 2.05, 0.0, 0.0, 0.6427876, 0.7660444,
            -1.95, 6.81, 2.05, 0.0, 0.0, -0.9659258, 0.258819
            ]
        }
        self.nextgate=2
        self.tr=Track("world/track.yaml")
        self.r=np.linalg.norm(self.tr.gates[1].pos)
        self.lastr=self.r
        self.env_train=True
        if 'env_train' in config:
            self.env_train=config['env_train']
        


        if 'physics' in config:
            for key, value in config['physics'].items():
                parameters_physics[key] = value

        self.physics_node = QuadrotorPybulletPhysics(suffix=f"{QuadrotorBaseEnv.num_envs}env_suffix", parameter_overrides=parameters_physics)

        if 'image' in sensors:
            parameters_camera = {
                'obstacles_description': ['gap', 'gap', 'gap', 'gap', 'gap', 'gap', 'gap'],
                'obstacles_poses': [
                    -0.6, -0.86, 3.68, 0.0, 0.0, -0.1736482, 0.9848078,
                    9.0, 6.45, 1.05, 0.0, 0.0, 0.0, 1.0,
                    8.85, -3.8, 1.05, 0.0, 0.0, -0.9063078, 0.4226183,
                    -4.3, -5.6, 3.4, 0.0, 0.0, -1.0, 0.0,
                    -4.3, -5.6, 1.42, 0.0, 0.0, 0.0, 1.0,
                    4.5, -0.45, 1.05, 0.0, 0.0, 0.6427876, 0.7660444,
                    -1.95, 6.81, 1.05, 0.0, 0.0, -0.9659258, 0.258819
                ]
            }
            if 'camera' in config:
                
                for key, value in config['camera'].items():
                    parameters_camera[key] = value
                    
            self.camera_node = QuadrotorPybulletCamera(suffix=env_suffix, parameters=parameters_camera)
        if 'imu' in sensors:
            parameters_imu ={}            

            if 'imu' in config:
                for key, value in config['imu'].items():
                    parameters_imu[key] = value
            self.imu_node = QuadrotorIMU(suffix=env_suffix, parameter_overrides=parameters_imu)
            self.imu = self.imu_node.imu

        self.ROT_HOVER_VEL = self.physics_node.ROT_HOVER_VEL
        self.ROT_MAX_VEL = self.physics_node.ROT_MAX_VEL
        self.M = self.physics_node.M
        self.W = self.physics_node.W
        self.MAX_THRUST = self.physics_node.MAX_THRUST
        self.J = self.physics_node.J
        self.KF=self.physics_node.KF
        self.KM=self.physics_node.KM
        self.ARM_X=self.physics_node.ARM_X
        self.ARM_Y=self.physics_node.ARM_Y


        self.action_space = gym.spaces.Box(low=np.array([0.0, -15.0, -15.0, -15.0]),
                                                       high=np.array([self.MAX_THRUST, 15.0, 15.0, 15.0]),
                                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)
        self.sensors = sensors
        self.state = self.physics_node.state


        self.time = 0
        self.time_limit = time_limit
        self.terminate_on_contact = terminate_on_contact

        self.workspace = np.array([[-2, 2], [-2, 2], [-2, 2]])
        self.velocity_space = np.array([[0, 15], [-15, 15], [-2, 2]])
        self.goal_pos = [0, 0, 0]
        self.goal_vel = [0, 0, 0]

        self.closed = False
        self.dt = self.physics_node.simulation_step_period
        self.t_lap=0
        QuadrotorBaseEnv.num_envs += 1
        

    def reset(self, *, seed=None, options=None):
        if self.closed:
            raise Exception("Trying to reset a closed environment")
        np.random.seed(seed)
        init_state = State()
        init_pose=False
        self.t_lap=0
        if(self.env_train):
            self.nextgate=np.random.randint(0,7)
            
            if self.nextgate==0:
                init_pose=True
                start_pos = [-4.0,3.0,1.0]
                gate_rotation= pyquaternion.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
                self.nextgate=1
            if self.nextgate==1:

                start_pos = self.tr.gates[6].pos
                gate_rotation=self.tr.gates[6].att
            else:
                start_pos = self.tr.gates[self.nextgate-2].pos
                gate_rotation=self.tr.gates[self.nextgate-2].att
        else:
            start_pos = [-4.0,3.0,1.0]
            gate_rotation= pyquaternion.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
            self.nextgate=1

        gate_rotation=Rotation.from_quat([gate_rotation.x,gate_rotation.y,gate_rotation.z,gate_rotation.w])
        gate_vector = gate_rotation.apply(np.array([1,0,0]))
        rand_deviation_z =np.random.uniform(-np.pi/4, np.pi/4)
        rand_deviation_y =np.random.uniform(-np.pi/4, np.pi/4)
        rotation_matrix_z = np.array([[np.cos(rand_deviation_z), -np.sin(rand_deviation_z), 0],
                [np.sin(rand_deviation_z), np.cos(rand_deviation_z), 0],
                [0, 0, 1]])
        rotation_matrix_y = np.array([[np.cos(rand_deviation_y), 0, np.sin(rand_deviation_y)],
                [0, 1, 0],
                [-np.sin(rand_deviation_y), 0, np.cos(rand_deviation_y)]])
        if (not (self.env_train)) or (init_pose):
            vel_vector=[0.0,0.0,0.0]
        else:
            vel_vector =np.random.uniform(2.0,8.0)* np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, gate_vector))

        init_state.pose.position.x = start_pos[0]+np.random.uniform(-0.5, 0.5)
        init_state.pose.position.y = start_pos[1]+np.random.uniform(-0.5, 0.5)
        init_state.pose.position.z = start_pos[2]+np.random.uniform(-0.5, 0.5)
        init_state.twist.linear.x =vel_vector[0]
        init_state.twist.linear.y = vel_vector[1]
        init_state.twist.linear.z = vel_vector[2]
        corners=self.tr.gates[self.nextgate-1].gate_corners([start_pos[0],start_pos[1],start_pos[2]],[0.0,0.0,0.0,1.0])

        center = np.mean(np.array(corners).reshape((4, 3)), axis=0)
        # Convert center to spherical coordinates
        self.r = np.linalg.norm(center)
        self.lastr=self.r
        self.last_action = [0.0, 0.0, 0.0, 0.0]
        
        # self.theta = np.arctan2(center[1], center[0])
        self.theta=self.tr.gates[self.nextgate-1].dev_angle([init_state.pose.orientation.x,init_state.pose.orientation.y,init_state.pose.orientation.z,init_state.pose.orientation.w])
        self.phi = np.arccos(center[2]/self.r)
        self.physics_node.set_state(init_state)        
        obs, _,_,_, info = self.step(np.array([0.0, 0.0, 0.0, 0.0]))
        self.time = 0
        self.closed = False
        info['t'] = self.time
        return obs, info

    def step(self, action):
        if self.closed:
            raise Exception("Trying to step in closed environment")
        self.time += self.dt
        action = np.array(action, dtype=float)
        self.physics_node.receive_thrust_bodyrates_commands_callback(action)
        obs = []
        self.state = self.physics_node.state
        pos=[self.state.pose.position.x,self.state.pose.position.y,self.state.pose.position.z]
        vel=[self.state.twist.linear.x,self.state.twist.linear.y,self.state.twist.linear.z]
        
        statedata = self.state
        reset_r=False
        passed=self.tr.gates[self.nextgate-1].is_passed(pos)
        if  not(passed==0):
            reset_r=True


            if(self.nextgate==7):
                self.nextgate=1
            else:
                self.nextgate=self.nextgate+1
        
                
        
            
        
        corners=self.tr.gates[self.nextgate-1].gate_corners([statedata.pose.position.x,statedata.pose.position.y,statedata.pose.position.z],[statedata.pose.orientation.x,statedata.pose.orientation.y,statedata.pose.orientation.z,statedata.pose.orientation.w])

        center = np.mean(np.array(corners).reshape((4, 3)), axis=0)
        # Convert center to spherical coordinates
        self.r = np.linalg.norm(center)
        # self.theta = np.arctan2(center[1], center[0])
        self.theta=self.tr.gates[self.nextgate-1].dev_angle([statedata.pose.orientation.x,statedata.pose.orientation.y,statedata.pose.orientation.z,statedata.pose.orientation.w])
        
        # get gate quaternion
        # gate_quat=self.tr.gates[self.nextgate-1].att()

        if not (self.r==0):
            self.phi = np.arccos(center[2]/(self.r))
        self.ang_acc=[statedata.accel.angular.x,statedata.accel.angular.y,statedata.accel.angular.z]
        if(reset_r):
            self.lastr=self.r
        
        if 'imu' in self.sensors:
            self.imu_node.receive_state_callback(self.state)
            self.imu_mesurements = np.array([self.imu.angular_velocity.x, self.imu.angular_velocity.y, self.imu.angular_velocity.z,self.imu.linear_acceleration.x, self.imu.linear_acceleration.y, self.imu.linear_acceleration.z])        
        obs=np.array([statedata.pose.position.x, statedata.pose.position.y, statedata.pose.position.z,
                                corners[0],corners[1],corners[2],
                                corners[3],corners[4],corners[5],
                                corners[6],corners[7],corners[8],
                                corners[9],corners[10],corners[11],
                                statedata.pose.orientation.x, statedata.pose.orientation.y, statedata.pose.orientation.z, statedata.pose.orientation.w,
                                statedata.twist.linear.x, statedata.twist.linear.y, statedata.twist.linear.z,
                                statedata.twist.angular.x, statedata.twist.angular.y, statedata.twist.angular.z])
# 
        info = {'t': self.time}
        contacted = (self.physics_node.check_contact())
        if contacted:
            info['contact'] = True
        truncated = False
        if self.time > self.time_limit and self.time_limit > 0:
            truncated = True
        terminated = False
        if self.terminate_on_contact and (contacted or obs[0]<-11.0 or  obs[0]>15.0 or obs[1]<-12.0 or obs[1]>12.0 or obs[2]>10.0 or obs[2]<=0.0):
            terminated = True
        

        reward = self.get_reward(obs,action)+passed
        if (self.env_train==False):
            self.t_lap=self.t_lap+1e-2


        
        

        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def get_reward(self, obs,action):
        lam_1=1.2
        lam_2=0.02
        lam_3=-10.0
        lam_4=-2e-4
        lam_5=-1e-4
        omega=obs[-3:]
        r_prog=lam_1*(self.lastr-self.r)
        diff_action=np.linalg.norm(np.array(action)-np.array(self.last_action))

        dev_ang=self.theta
        r_perc=lam_2*np.exp(lam_3*dev_ang**4)
        r_cmd=lam_4*np.linalg.norm(omega)+lam_5*diff_action
        if(self.physics_node.check_contact() ) :
            r_coll=-5.0
           
        else:
            r_coll=0
        self.lastr=self.r
        reward=r_prog +r_cmd+r_coll +r_perc
        return reward

    def get_images(self,state):
        self.camera_node.receive_state_callback(state)
        image=self.camera_node.cv_image
        return image
        

    def close(self):
        print("Closing")
        if not self.closed:
            QuadrotorBaseEnv.num_envs -= 1
            self.physics_node.destroy_node()
            if 'image' in self.sensors:
                self.camera_node.destroy_node()
            self.closed = True

    def __del__(self):
        self.close()
