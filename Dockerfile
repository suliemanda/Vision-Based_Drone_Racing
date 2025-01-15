FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3-pip \
    xauth 

RUN pip install numpy==1.24.4 \
                scipy==1.10.1 \
                sympy==1.12 \
                matplotlib==3.7.4 \
                gymnasium==0.29.1 \
                ipython==8.12.3 \
                opencv_python==4.8.1.78 \
                pandas==2.0.3 \
                pybullet==3.2.5 \
                PyQt5==5.15.5 \
                PyYAML==5.3.1 \
                pytransform3d==3.5.0 \
                stable_baselines3==2.2.1 \
                torch==2.4.0\
                torchvision==0.19.0\
                pytorch_lightning==1.7.1 \
                lightly\
                pytorch-tcn
                


RUN pip install tensorboard \
                tqdm \
                rich

