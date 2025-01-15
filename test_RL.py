
from src._base import QuadrotorBaseEnv
import time
import numpy as np
import stable_baselines3 as sb
from stable_baselines3.common.policies import ActorCriticPolicy,BasePolicy
import numpy as np
import torch as th
from gymnasium import spaces
from src.types import State
import torch 
import cv2
policy_kwargs = dict(activation_fn=lambda: torch.nn.LeakyReLU(0.2),
                         net_arch=[128, 128]
                         )


ep_steps = 620
dt = 1/100
ep_time = ep_steps * dt
ep_steps = int(int(ep_time)/dt)

server = 'DIRECT'

env = QuadrotorBaseEnv(env_suffix='',
                       sensors=['image'],
                       terminate_on_contact=True,
                       time_limit=int(ep_time),
                       config={'physics': {'physics_server': server, 'quadrotor_description': 'racing', 'render_ground': False,'sequential_mode':True, 'publish_state': True,'cmd_type':'thrust_bodyrates'},'camera':{'render_architecture':True, 'camera_fov':120.0,'render_ground':False,'quadrotor_description': 'racing'}, 'env_train' : False}
                       )


model = sb.PPO.load("results/models/racing_lr34_entcoef53_sched_envs10_bodyrates_con")

obs_space=model.observation_space
act_space=model.action_space
lr_sch=model.lr_schedule


num_exp = 1
total_rewards = []
t=0
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
for _ in range(num_exp):
    total_reward = 0
    obs, info = env.reset()


    while True:
        action, _states = model.predict(obs, deterministic=True)
        


        obs,reward, terminated, truncated, info = env.step(action)
        state=State()
        p=obs[0:3]
        q=obs[15:19]
        state.pose.position.x=p[0]
        state.pose.position.y=p[1]
        state.pose.position.z=p[2]
        state.pose.orientation.x=q[0]
        state.pose.orientation.y=q[1]
        state.pose.orientation.z=q[2]
        state.pose.orientation.w=q[3]
        env.camera_node.receive_state_callback(state)
        image=env.camera_node.cv_image
        cv2.imshow('image',image)
        cv2.waitKey(1)
        t=t+1
        
        total_reward += reward
        if terminated or truncated:
            break
    total_rewards.append(total_reward)

    print(total_reward)

print(f"On average, this agent gets {np.mean(total_rewards)} reward per episode")
