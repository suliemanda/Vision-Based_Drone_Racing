#!/usr/bin/env python3

import pybullet as p
import pybullet_data
from .types import State
import cv2
import xacro
import os
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import IPython.core.ultratb
except ImportError:
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()

DEFAULT_FREQUENCY_IMG = 30  # Hz
DEFAULT_QOS_PROFILE = 10


class QuadrotorPybulletCamera():

    def __init__(self, suffix='', parameters={}):
        """ Initializes the node."""
        self.physics_server=parameters.get('physics_server', 'GUI')
        self.quadrotor_description_file_name=parameters.get('quadrotor_description', 'cf2x')
        self.quadrotor_scale=parameters.get('quadrotor_scale', 1.0)
        self.obstacles_description_file_names=parameters.get('obstacles_description', ['NONE'])
        self.obstacles_poses=parameters.get('obstacles_poses', [0.0])
        self.render_ground=parameters.get('render_ground', True)
        self.render_architecture=parameters.get('render_architecture', False)
        self.state_topic=parameters.get('state_topic', 'quadrotor_state'+suffix)
        self.image_topic=parameters.get('image_topic', 'quadrotor_img'+suffix)
        self.image_width=parameters.get('image_width', 800)
        self.image_height=parameters.get('image_height', 600)
        self.camera_position=parameters.get('camera_position', [0.0, 0.0, 0.0])
        self.camera_main_axis=parameters.get('camera_main_axis', [1.0, 0.0, 0.0])
        self.camera_up_axis=parameters.get('camera_up_axis', [0.0, 0.0, 1.0])
        self.camera_focus_distance=parameters.get('camera_focus_distance', 1.0)
        self.camera_fov=parameters.get('camera_fov', 60.0)
        self.camera_near_plane=parameters.get('camera_near_plane', 0.01)
        self.camera_far_plane=parameters.get('camera_far_plane', 100.0)
        self.sequential_mode=parameters.get('sequential_mode', True)
        self.publish_image=parameters.get('publish_image', False)       


        # initialize the constants, the urdf file and the pybullet client
        self.initialize_urdf()
        self.initialize_pybullet()

        # Initialize the published and received data
        self.initialize_data()
        # Announce that the node is initialized
   

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
        world_description_folder='world'
        if (self.physics_server == 'DIRECT'):
            self.physicsClient = p.connect(p.DIRECT)
        elif (self.physics_server == 'SHARED_MEMORY'):
            self.physicsClient = p.connect(p.SHARED_MEMORY)
        else:
            self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if (self.render_ground):
            self.planeId = p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)
        if (self.render_architecture):
            p.loadSDF(os.path.join(world_description_folder,"basket_l.sdf"), physicsClientId=self.physicsClient)
        self.obstacleIds = []
        for (i, obstacle_urdf_file) in enumerate(self.obstacle_urdf_files):
            print(i,obstacle_urdf_file)
            self.obstacleIds.append(p.loadURDF(
                obstacle_urdf_file, self.obstacles_poses[i*7: i*7+3], self.obstacles_poses[i*7+3: i*7+7], useFixedBase=True, physicsClientId=self.physicsClient))
        self.quadrotor_id = p.loadURDF(self.quadrotor_urdf_file, [-4.0,3.0,1.0], globalScaling=self.quadrotor_scale, physicsClientId=self.physicsClient)
        # Configure the debug visualizer
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physicsClient)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self.physicsClient)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.physicsClient)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self.physicsClient)

    def initialize_data(self):
        self.state = State()
        self.cv_image = None    

    def receive_state_callback(self, msg):
        self.state = msg
        if self.sequential_mode:
            self.publish_image_callback()

    def publish_image_callback(self) -> None:

        self.publish_image_callback_thread()

    def publish_image_callback_thread(self):
        pose = self.state.pose
        quad_pos = [pose.position.x, pose.position.y, pose.position.z]
        quad_quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        p.resetBasePositionAndOrientation(self.quadrotor_id, quad_pos, quad_quat, physicsClientId=self.physicsClient)
        rot_quat = Rotation.from_quat(quad_quat)
        init_camera_vector = np.array(self.camera_main_axis)/np.linalg.norm(self.camera_main_axis)
        init_up_vector = np.array(self.camera_up_axis)/np.linalg.norm(self.camera_up_axis)
        # # Rotated vectors
        camera_vector = rot_quat.apply(init_camera_vector)
        up_vector = rot_quat.apply(init_up_vector)

        camera_pos = rot_quat.apply(np.array(self.camera_position)) + np.array(quad_pos)
        camera_focus_target = camera_pos + camera_vector*self.camera_focus_distance

        view_matrix = p.computeViewMatrix(
            camera_pos, camera_focus_target, up_vector)

        projection_matrix = p.computeProjectionMatrixFOV(fov=self.camera_fov, aspect=float(
            self.image_width) / self.image_height, nearVal=self.camera_near_plane, farVal=self.camera_far_plane)

        _, _, px, _, _ = p.getCameraImage(
            width=self.image_width,
            height=self.image_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            flags=p.ER_NO_SEGMENTATION_MASK,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.physicsClient
        )

        image_rgb = np.array(px, dtype=np.uint8)
        image_rgb = np.reshape(image_rgb, (self.image_height, self.image_width, 4))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGR)
        self.cv_image =image_bgr
    def destroy_node(self):
        p.disconnect(self.physicsClient)
        cv2.destroyAllWindows()
        print('Node destroyed')
        

