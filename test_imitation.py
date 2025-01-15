
from src._base import QuadrotorBaseEnv
import numpy as np
import stable_baselines3 as sb
import torch
import copy
import numpy as np
from torch import nn
import torchvision
from src.types import  State
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import argparse

from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad
from pytorch_tcn import TCN
from PIL import Image
import cv2

from lightly.transforms.byol_transform import BYOLView1Transform
policy_kwargs = dict(activation_fn=lambda: torch.nn.LeakyReLU(0.2),
                         net_arch=[128, 128]
                         )
class QuadrotorModelEarlyConcat(nn.Module):
    def __init__(
        self, image_backbone, action_dim, imu_channels, image_feature_dim
    ):
        super(QuadrotorModelEarlyConcat, self,).__init__()

        # CNN for images
        self.cnn = image_backbone
        self.cnn.eval()
        self.cnn.requires_grad = False

        self.V_tcn = TCN(
            num_inputs=image_feature_dim,
            num_channels=[64,128, 64],
            kernel_size=12,
            dropout=0.2,
            activation='gelu'
        )
        self.IMU_tcn = TCN(num_inputs=imu_channels, num_channels=[64, 128, 64], kernel_size=12, dropout=0.2, activation='gelu')
        
        self.mlp_head = nn.Sequential(
            nn.Linear(128, 128), nn.LeakyReLU(0.2), nn.Linear(128, action_dim)
        )
        self.transform =BYOLView1Transform(input_size=256,
        hf_prob=0.0,
        normalize=None,
        cj_prob=0.0,
        cj_strength=0.0,
        cj_bright=0.0,
        cj_contrast=0.0,
        cj_sat=0.0,
        cj_hue=0.0,
        min_scale=1.0,
        random_gray_scale=0.0,
        gaussian_blur=0.0)
    
    
    def transformer(self,images):
    #    im_pil = Image.fromarray(images)

       return self.transform(images)



    def forward(self, images=None,imu=None ): #imu_seq
        # Extract image features
        if not (images==None):
            self.image_features = self.cnn(images) 
            self.im_tcn_output = self.V_tcn(self.image_features.T).T
        else:
            self.im_tcn_output=self.im_tcn_output.detach()

        if not(imu==None):
            self.imu=imu
        
        imu_output = self.IMU_tcn(self.imu.T).T

        
        tcn_output = torch.cat((self.im_tcn_output, imu_output), dim=1)
        actions = self.mlp_head(tcn_output)

        return tcn_output, actions


class BYOL(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(2048, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 512, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)
        deactivate_requires_grad(self.backbone)
        deactivate_requires_grad(self.projection_head)
        deactivate_requires_grad(self.prediction_head)

        

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        z = z.detach()
        # y=y.detach()
        return z


COS=torch.nn.CosineSimilarity(eps=1e-4)
MSE=torch.nn.MSELoss()
def behavior_cloning_loss(
     actions, expert_actions
):
    

    # Loss for actions
    action_loss = MSE(actions,expert_actions)
    sim_loss=(1-COS(actions,expert_actions))
    return action_loss, sim_loss


def get_env():
    env = QuadrotorBaseEnv(env_suffix='',
                           sensors=['image','imu'],
                           terminate_on_contact=True,
                           time_limit=6,
                           config={'physics': {'physics_server': 'DIRECT', 'quadrotor_description': 'racing', 'render_ground': False},'camera':{'threading':False,'render_architecture':True, 'camera_fov':120.0,'render_ground':False,'quadrotor_description': 'racing','physics_server': 'GUI'}, 'env_train' : False}
                           )
    return env
def main():
    num_env=1
    parser = argparse.ArgumentParser(description='Process Learning Parameters')
    parser.add_argument('-ls', dest='model_path_load', help='Path to the model to load', type=str, default=None)
    parser.add_argument('-lt', dest='teacher_model_path_load', help='Path to the teacher model to load', type=str, default=None)


    args=parser.parse_args()
    model_path_load=args.model_path_load

    ep_steps = 600
    dt = 1/100
    ep_time = ep_steps * dt
    ep_steps = 1000#int(int(ep_time)/dt)


    train_env =get_env()# make_vec_env(get_env, num_env, vec_env_cls=vec_cls)

    teacher_path=args.teacher_model_path_load
    if(teacher_path==None):
        raise ValueError("Teacher model path not provided, use -lt to provide the path to the teacher model")
    model = sb.PPO.load(teacher_path)

    resnet = torchvision.models.resnet50()

    backbone = nn.Sequential(*list(resnet.children())[:-1])
    feature_extactor = BYOL(backbone)
    student_model = QuadrotorModelEarlyConcat(
        image_backbone=feature_extactor,
        action_dim=4,
        imu_channels=6,
        image_feature_dim=256,

    )


    if(model_path_load!=None):
        student_model.load_state_dict(torch.load(model_path_load,weights_only=True))
    else:
        raise ValueError("Student model path not provided, use -ls to provide the path to the teacher model")

    student_model.eval()
  
    student_model.to("cuda")

    os.system("clear")
    ep=0.3    

    for _ in range(0,1,1):
        ep_len=0
        loss=0
        DAloss=0
        DAseq_loss=0
        avg_simloss=0
        avg_actloss=0
        rat=0
        obs, info = train_env.reset()
        p=obs[0:3]
        q=obs[15:19]
        state=State()
        state.pose.position.x=p[0]
        state.pose.position.y=p[1]
        state.pose.position.z=p[2]
        state.pose.orientation.x=q[0]
        state.pose.orientation.y=q[1]
        state.pose.orientation.z=q[2]
        state.pose.orientation.w=q[3]
        student_model.V_tcn.reset_buffers()
        student_model.IMU_tcn.reset_buffers()
        sequence_loss=0
        avg_sequence_loss=0
        t=3
        st=0
      
        
        while True:
            
            p=obs[0:3]
            q=obs[15:19]
            imu_mes=train_env.imu_mesurements


            state=State()
            state.pose.position.x=p[0]
            state.pose.position.y=p[1]
            state.pose.position.z=p[2]
            state.pose.orientation.x=q[0]
            state.pose.orientation.y=q[1]
            state.pose.orientation.z=q[2]
            state.pose.orientation.w=q[3]
            imu_mes=imu_mes.reshape(num_env,6)
            imu_mes = torch.tensor(imu_mes, dtype=torch.float32).to("cuda")        
            if(t==3):

                im=train_env.get_images(state)
                cv2.imshow("image",im)
                cv2.waitKey(5)
                im = Image.fromarray(im)


                im=student_model.transformer(im)
                im=im.to("cuda")
                im=im.unsqueeze(0)
                tcn_output, pred_action=student_model.forward(images=im,imu=imu_mes)
                t=0

            else:
                tcn_output, pred_action=student_model.forward(imu=imu_mes)

            t=t+1


            if(st>12*3):
                action,__= model.predict(obs,deterministic=True)
                
                
                if((np.random.random()>ep) ): 
                    rat+=1
                    action=pred_action.detach().cpu().numpy().flatten()
                obs,reward, terminated, truncated, info = train_env.step(action)


            else:
                action=np.array([train_env.M*9.82,0,0,0])
                terminated=False
                truncated=False
 
            ep_len+=1

            if terminated or truncated:
                break
            st+=1





if __name__=="__main__":
    main()