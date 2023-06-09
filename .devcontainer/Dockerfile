# FROM tensorflow/tensorflow
FROM tensorflow/tensorflow:latest-gpu
#nvidia/cuda:12.1.1-runtime-ubuntu22.04
RUN echo "Europe/Utc" > /etc/timezone
# RUN ln -fs /usr/share/zoneinfo/Europe/Rome /etc/localtime

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
# RUN apt-get update && apt-get upgrade -y
# RUN apt-get install -y python3-pip
ENV SHELL /bin/bash

RUN apt-get update -q && \
	export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends tzdata
RUN dpkg-reconfigure -f noninteractive tzdata
# Install packages
RUN apt-get update -q && \
	export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends apt-utils software-properties-common wget curl rsync netcat mg vim bzip2 zip unzip && \
    apt-get install -y --no-install-recommends libxtst6 && \
    apt-get install -y --no-install-recommends git && \
    apt-get install -y --no-install-recommends bash-completion && \
    apt-get install -y --no-install-recommends nano && \
    apt-get install -y --no-install-recommends neovim && \
    apt-get install -y --no-install-recommends net-tools && \
    apt-get install -y --no-install-recommends iputils-ping && \
    apt-get install -y --no-install-recommends terminator && \
    apt-get autoclean -y && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update -q && \
        export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y --no-install-recommends install libgl1-mesa-glx libgl1-mesa-dri && \
    apt-get install -y mesa-utils && \
    rm -rf /var/lib/apt/lists/*
RUN sed -i 's/--no-generate//g' /usr/share/bash-completion/completions/apt-get && \
    sed -i 's/--no-generate//g' /usr/share/bash-completion/completions/apt-cache
WORKDIR /home/workspace

RUN sed -i "s/#force_color_prompt=yes/force_color_prompt=yes/g" /root/.bashrc

RUN touch /root/.Xauthority

# RUN sudo apt update && sudo apt upgrade -y && sudo apt install python3-venv -y

# RUN sudo apt update && sudo apt upgrade -y  && \
#     sudo add-apt-repository -y ppa:deadsnakes/ppa && \
#     sudo apt update && \
#     sudo apt install python3.9 -y  && \
#     sudo apt install python3.9-venv -y && \ 
#     curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py  && \
#     python3.9 get-pip.py && \
#     rm get-pip.py
    
# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
CMD ["/bin/bash"]
