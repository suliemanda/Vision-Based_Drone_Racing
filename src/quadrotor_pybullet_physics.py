import os
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
import pybullet as p
import pybullet_data
import xacro
from .types import State

try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()


DEFAULT_FREQUENCY = 240  # Hz
DEFAULT_QOS_PROFILE = 10


class QuadrotorPybulletPhysics():

    def __init__(self, suffix='', parameter_overrides={}):
        """ Initializes the node."""
        self.physics_server = parameter_overrides.get('physics_server', 'GUI')
        self.quadrotor_description_file_name = parameter_overrides.get('quadrotor_description', 'racing')
        self.obstacles_description_file_names = parameter_overrides.get('obstacles_description', ['NONE'])
        self.obstacles_poses = parameter_overrides.get('obstacles_poses', [0.0])
        self.render_ground = parameter_overrides.get('render_ground', False)
        self.simulation_step_frequency = parameter_overrides.get('simulation_step_frequency', DEFAULT_FREQUENCY)


        # Control the frequencies of simulation
        self.simulation_step_period = 1.0 / self.simulation_step_frequency  # seconds

        # initialize the constants, the urdf file and the pybullet client
        self.initialize_urdf()
        self.initialize_constants()
        self.initialize_pybullet()

        # Initialize the published and received data
        self.initialize_data()
        init_state= State()
        start_pos=[-4.0,3.0,1.0]
        init_state.pose.position.x = start_pos[0]
        init_state.pose.position.y = start_pos[1]
        init_state.pose.position.z = start_pos[2]
        self.set_state(init_state)


    def initialize_constants(self):
        config_folder =os.path.join('quadrotor_description', 'config') 
        config_file = os.path.join(config_folder, self.quadrotor_description_file_name+'_params.yaml')
        with open(config_file, "r") as stream:
            try:
                parameters = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                parameters = dict()

        quadrotor_params = parameters[f'{self.quadrotor_description_file_name.upper()}_PARAMS']
        self.G = 9.81
        self.KF = quadrotor_params['KF']
        self.KM = quadrotor_params['KM']
        self.M = quadrotor_params['M']
        self.W = self.M*self.G
        self.ROT_HOVER_VEL = np.sqrt(self.W/(4*self.KF))
        self.T2W = quadrotor_params['T2W']
        self.ROT_MAX_VEL = np.sqrt(self.T2W*self.W/(4*self.KF))
        self.ROT_MAX_ACC = quadrotor_params['ROT_MAX_ACC']
        self.ROT_TIME_STEP = quadrotor_params['ROT_TIME_STEP']
        self.DRAG_MAT_LIN = np.array(quadrotor_params['DRAG_MAT_LIN'])
        self.DRAG_MAT_QUAD = np.array(quadrotor_params['DRAG_MAT_QUAD'])
        self.ROTOR_DIRS = quadrotor_params['ROTOR_DIRS']
        self.ARM_X = quadrotor_params['ARM_X']
        self.ARM_Y = quadrotor_params['ARM_Y']
        self.ARM_Z = quadrotor_params['ARM_Z']
        self.J = np.array(quadrotor_params['J'])
        self.MAX_THRUST = self.M*self.G*self.T2W
        self.MAX_INDIVIDUAL_THRUST = self.MAX_THRUST/4

  

    def initialize_urdf(self):
        quadrotor_description_folder = os.path.join('quadrotor_description','description')
        quadrotor_description_file = os.path.join(quadrotor_description_folder, self.quadrotor_description_file_name+'.urdf.xacro')
        quadrotor_description_content = xacro.process_file(quadrotor_description_file).toxml()
        new_file = os.path.join(quadrotor_description_folder, self.quadrotor_description_file_name+'.urdf')
        with open(new_file, 'w+') as f:
            f.write(quadrotor_description_content)
        self.quadrotor_urdf_file = new_file

        # Retreive the obstacle urdf file and save it for pybullet to read
        obstacles_description_folder = 'world'
        self.obstacle_urdf_files = []
        for name in self.obstacles_description_file_names:
            if (name == 'NONE'):
                break
            self.obstacle_description_file_name = name
            obstacle_description_file = os.path.join(obstacles_description_folder, self.obstacle_description_file_name+'.urdf.xacro')
            obstacle_description_content = xacro.process_file(obstacle_description_file).toxml()
            new_file = os.path.join(obstacles_description_folder, name + '.urdf')
            with open(new_file, 'w+') as f:
                f.write(obstacle_description_content)
            self.obstacle_urdf_files.append(new_file)
        

    def initialize_pybullet(self):
        server_map = {"DIRECT": p.DIRECT,
                      "GUI": p.GUI,
                      "SHARED_MEMORY": p.SHARED_MEMORY,
                      "GUI_SERVER": p.GUI_SERVER}
        try:
            self.physicsClient = p.connect(server_map[self.physics_server])
        except Exception as e:
            self.destroy_node()
            raise e

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.simulation_step_period, physicsClientId=self.physicsClient)
        p.setGravity(0, 0, -self.G, physicsClientId=self.physicsClient)
        if (self.render_ground):
            world_description_folder=os.path.join('src', 'world')
            p.loadSDF(os.path.join(world_description_folder,"basket_l.sdf"), physicsClientId=self.physicsClient)

        self.obstacleIds = []
        for (i, obstacle_urdf_file) in enumerate(self.obstacle_urdf_files):
            self.obstacleIds.append(p.loadURDF(
                obstacle_urdf_file, self.obstacles_poses[i*7: i*7+3], self.obstacles_poses[i*7+3: i*7+7], useFixedBase=True, physicsClientId=self.physicsClient))
        self.quadrotor_id = p.loadURDF(self.quadrotor_urdf_file, [-4.0,3.0,1.0], flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=self.physicsClient)
        
        # Disable default damping of pybullet!
        p.changeDynamics(self.quadrotor_id, -1, linearDamping=0, angularDamping=0, physicsClientId=self.physicsClient)
        p.changeDynamics(self.quadrotor_id, 0, linearDamping=0, angularDamping=0, physicsClientId=self.physicsClient)
        p.changeDynamics(self.quadrotor_id, 1, linearDamping=0, angularDamping=0, physicsClientId=self.physicsClient)
        p.changeDynamics(self.quadrotor_id, 2, linearDamping=0, angularDamping=0, physicsClientId=self.physicsClient)
        p.changeDynamics(self.quadrotor_id, 3, linearDamping=0, angularDamping=0, physicsClientId=self.physicsClient)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physicsClient)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self.physicsClient)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.physicsClient)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self.physicsClient)

    def initialize_data(self):
        self.thrust_bodyrates=np.array([self.W,0,0,0])
        self.state = State()
        self.state.pose.position.z = 0.25
       

    def set_state(self, msg: State):
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                        msg.pose.orientation.z, msg.pose.orientation.w])
        v = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        w = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])
        w_W = Rotation.from_quat(quat).apply(w)
        p.resetBasePositionAndOrientation(self.quadrotor_id, pos, quat, physicsClientId=self.physicsClient)
        p.resetBaseVelocity(self.quadrotor_id, v, w_W, physicsClientId=self.physicsClient)

    def receive_thrust_bodyrates_commands_callback(self, msg):
        self.thrust_bodyrates = msg
      
        self.simulation_step_callback()




    def apply_simulation_step(self):
        pos0, quat0 = p.getBasePositionAndOrientation(self.quadrotor_id, physicsClientId=self.physicsClient)
        pos0, quat0 = np.array(pos0), np.array(quat0)
        vel0, avel0_W = p.getBaseVelocity(self.quadrotor_id, physicsClientId=self.physicsClient)
        vel0, avel0_B = np.array(vel0), Rotation.from_quat(quat0).inv().apply(np.array(avel0_W))
        p.stepSimulation(physicsClientId=self.physicsClient)
        pos, quat = p.getBasePositionAndOrientation(self.quadrotor_id, physicsClientId=self.physicsClient)
        pos, quat = np.array(pos), np.array(quat)
        vel, avel_W = p.getBaseVelocity(self.quadrotor_id, physicsClientId=self.physicsClient)
        vel, avel_B = np.array(vel), Rotation.from_quat(quat).inv().apply(np.array(avel_W))
        accel, anaccel = (vel-vel0)/self.simulation_step_period, (avel_B-avel0_B)/self.simulation_step_period

        self.state.pose.position.x = pos0[0]
        self.state.pose.position.y = pos0[1]
        self.state.pose.position.z = pos0[2]
        self.state.pose.orientation.x = quat0[0]
        self.state.pose.orientation.y = quat0[1]
        self.state.pose.orientation.z = quat0[2]
        self.state.pose.orientation.w = quat0[3]
        self.state.twist.linear.x = vel0[0]
        self.state.twist.linear.y = vel0[1]
        self.state.twist.linear.z = vel0[2]
        self.state.twist.angular.x = avel0_B[0]
        self.state.twist.angular.y = avel0_B[1]
        self.state.twist.angular.z = avel0_B[2]
        self.state.accel.linear.x = accel[0]
        self.state.accel.linear.y = accel[1]
        self.state.accel.linear.z = accel[2]
        self.state.accel.angular.x = anaccel[0]
        self.state.accel.angular.y = anaccel[1]
        self.state.accel.angular.z = anaccel[2]
        self.state.quadrotor_id = self.quadrotor_id

    def simulation_step_callback(self):
        thrust=self.thrust_bodyrates[0]
        w_x =self.thrust_bodyrates[1]
        w_y=self.thrust_bodyrates[2]
        w_z=self.thrust_bodyrates[3]
        self.apply_bodyrates(thrust,w_x,w_y,w_z)
        self.apply_simulation_step()
       
    def apply_bodyrates(self,thrust,w_x,w_y,w_z):
        force = np.array([0,0,thrust])
        w_B=[w_x,w_y,w_z]
        quad_quat = np.array([self.state.pose.orientation.x, self.state.pose.orientation.y,
                              self.state.pose.orientation.z, self.state.pose.orientation.w])
        # to world frame
        R = Rotation.from_quat(quad_quat)
        w_w = R.apply(w_B)

        p.resetBaseVelocity(self.quadrotor_id,angularVelocity=w_w, physicsClientId=self.physicsClient)
        p.applyExternalForce(self.quadrotor_id, -1, forceObj=force, posObj=[0, 0, 0], flags=p.LINK_FRAME, physicsClientId=self.physicsClient)
        

    def check_contact(self):
        if len(p.getContactPoints(self.quadrotor_id, physicsClientId=self.physicsClient)) > 0:
            return True
        return False
    def destroy_node(self):
        p.disconnect(physicsClientId=self.physicsClient)
        print('Node destroyed')
