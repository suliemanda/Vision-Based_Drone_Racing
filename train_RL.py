from src._base import QuadrotorBaseEnv
from typing import Callable
import torch
import stable_baselines3 as sb
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import argparse
import os
from stable_baselines3.common.callbacks import  CheckpointCallback



def get_env(server,env_train):
    env = QuadrotorBaseEnv(sensors=[],terminate_on_contact=True,
                           config={'physics': {'physics_server': server, 'quadrotor_description': 'racing', 'render_ground': False}, 'env_train' : env_train},
                           )
    return env


def main():
    parser = argparse.ArgumentParser(description='Process Learning Parameters')
    parser.add_argument('-e', dest='envs', help='Number of environment to use for collecting data', type=int, default=10)
    parser.add_argument('-m', dest='multi_processing', help='If set, use multi processing', action='store_true')
    parser.add_argument('-l', dest='model_path_load', help='Path to the model to load', type=str, default=None)
    parser.add_argument('-s', dest='model_path_save', help='Directory to save the model', type=str, default='./results/models')
    parser.add_argument('-v', dest='verbose', help='Whether to print the logs', action='store_true')
    parser.add_argument('-t', dest='tensorboard_log', help='Path to the tensorboard log', type=str, default='./results/logs/')
    parser.add_argument('-r', dest='render', help='Whether to render at least one GUI, for multi_processing this means rendering all environments', action='store_true')
    parser.add_argument('-n', dest='name', help='Name of the experiment', type=str, default='ppo_quadrotor')
    args = parser.parse_args()
    num_envs = args.envs
    vec_cls = DummyVecEnv if not args.multi_processing else SubprocVecEnv
    server = 'GUI' if args.render else 'DIRECT'
    train_env = make_vec_env(get_env, num_envs, env_kwargs={'server': server ,'env_train':True }, vec_env_cls=vec_cls)
    
    policy_kwargs = dict(activation_fn=lambda: torch.nn.LeakyReLU(0.2),
                         net_arch=[128, 128]
                         )
    roll_ep = 1
    ep_steps = 1500
    n_steps = roll_ep * ep_steps
    batch_size = n_steps * num_envs

    model_name = args.name
    tensorboard_log = args.tensorboard_log
    tensorboard_log = os.path.join(tensorboard_log, model_name)
    verbose = args.verbose
    intit_lr=3e-4
    entropy_c=5e-3
    save_dir = args.model_path_save
    check_point=CheckpointCallback(save_freq=1000000,save_path=save_dir,name_prefix=f'racing_lr{34}_entcoef{53}_sched_envs{num_envs}_bodyrates_new')
    

    def linear_schedule(initial_value: float) -> Callable[[float], float]:
            
        def func(progress_remaining: float) -> float:
            
            if(progress_remaining>0.2):
                return progress_remaining* initial_value
            else:
                return 0.2* initial_value


        return func
    model = sb.PPO("MlpPolicy", train_env, n_steps=n_steps, batch_size=batch_size, verbose=verbose,
                   policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=linear_schedule(intit_lr),ent_coef=entropy_c)
    
    load_file = args.model_path_load
    if load_file is not None:
        model.set_parameters(load_file)
    learn_steps = int(2e8)
    rend_st=ep_steps*800
    ep=int(learn_steps/rend_st)
    try:
        model.learn(total_timesteps=learn_steps, progress_bar=True, log_interval=1,reset_num_timesteps=True,callback=check_point)

    except KeyboardInterrupt:
        pass
    
    save_dir = os.path.join(save_dir, model_name)
    model.save(save_dir)


if __name__ == '__main__':
    main()
