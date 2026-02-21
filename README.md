# 🧮 Unparallel Parallelism

Welcome to **Unparallel Parallelism**, a fun workshop exploring the quirks and wonders of parallel computing with CUDA and GPUs.  

Here, we’ll go from “what is a thread?” to “how do I make thousands of them work… or misbehave?”  


### ⚠️ Disclaimer

No GPUs were harmed in the making of this workshop. Any kernel panics are purely educational.

## 💻 Prerequisites
- 🐧 Windows or Linux machine with NVIDIA GPU. Below instruction are meant for Windows machine but you can do the same in Linux machine.
- ⛔ MacOS not supported by CUDA


-------------------------------------------------------------------------------------------------------------------------------------------------

# ⚙️ **SETUP**

### 🐧 Install [`WSL2`](https://learn.microsoft.com/en-us/windows/wsl/) and setup Ubuntu on the Host Windows Machine
> This will give you a Linux environment where Docker + NVIDIA GPU works.
  - 🐧 VERSION 2 → WSL2 (required for GPU support).
  - ⛔ VERSION 1 → WSL1 (no GPU support).

```bash
    # Install Ubuntu and give username pass when asked eg: (hitesh/root)
    wsl --install -d Ubuntu-22.04

    # Check available Installed Destro
    wsl --list --verbose

    # Set to WSL 2
    wsl --set-default-version 2
    wsl --set-version <Distribution Name>  2
    wsl --set-version   Ubuntu-22.04 2

    #  Start Destro
    wsl --distribution <Distribution Name> --user <User Name>
    wsl --distribution  Ubuntu-22.04 --user hitesh

    wsl --terminate <Distribution Name>
    wsl --terminate Ubuntu-22.04

```

- Update your WSL2 Ubuntu

```bash
    sudo apt update
    sudo apt upgrade -y

    #  Install Utils
    sudo apt install -y apt-transport-https ca-certificates curl software-properties-common gnupg lsb-release

```

### 🎮 NVIDIA GPU drivers for WSL Linux on the Host Windows Machine
Download NVIDIA drivers for WSL: [https://developer.nvidia.com/cuda/wsl](https://developer.nvidia.com/cuda/wsl)

After installation, check on host machine:

```bash

    nvidia-smi

    // Output should look like this

    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 560.35.02              Driver Version: 560.94         CUDA Version: 12.6     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    Segmentation fault (core dumped)

```

### 🐳 Install Docker on WSL2 Linux

 Add Docker’s official repository

    ```bash

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    ```

- Install Docker Engine

    ```bash
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io
    ```

- Add your user to the Docker group

    ```bash
    sudo groupadd docker        # only needed if the group doesn't exist
    sudo usermod -aG docker $USER
    ```

- After this, you need to reload your group membership:
    > newgrp docker

Lastly Test Docker Installation on WSL2:

```bash
    docker --version

    # Starts Docker ::  must be on WSL2 
    sudo service docker start

    #  Run test container
    docker run hello-world

```

### 🌀 Enable GPU support inside WSL2 

Test GPU passthrough with [**Official CUDA Docker Image**](https://hub.docker.com/r/nvidia/cuda/tags):

> docker run --gpus all nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04 nvidia-smi

## 📟 Install CUDA CLI inside WSL2 Linux Ubuntu 

- First you will need `gcc` in WSL2 to compile CUDA source

```bash
    # Install gcc
    sudo apt install gcc
    sudo apt-get install g++

    # Verify
    gcc --version 

```

Follow the instructions to download [Installer of Linux WSL- Ubuntu 2.0 X86_64](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local)

- It will take some time to download Cuda source

```bash
    wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_580.95.05_linux.run

```

- lastly run the Cuda installer:

    ```bash
    sudo sh cuda_13.0.2_580.95.05_linux.run
    ```

- Add to `$PATH`

```bash
    echo $SHELL

    # Output: /bin/bash or /bin/zsh

    #  If it says bin/bash, add the following lines to the ~/.bashrc rile. 
    #  If it says bin/zsh,  add to the ~/.zshrc file.

    echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc

```

Validate Installation to get info about the **Nvidia CUDA compiler **(`nvcc`) & connected GPU **System Management Interface**(`smi`)

```bash
    nvcc --version
    nvidia-smi

```

Congragulations now you are ready to compile CUDA code

### Note ::  Using shell Script 📝
One shot approach to setup everything with help of script.

- Install WSL and Linux from Prerequisites
- Install rest of tools with helper script

    ```bash
    # start WSL
    wsl

    # Mount script
    cd /mnt/d/GitHub/unparallel-parallelism/scripts
    
    # Run setp script
    chmod +x run_me_first.sh
    ./run_me_first.sh

    ```
----------------------

# ▶️ **RUN THE PROJECT**

## 1. Run with WSL locally

### Start WSL 2.0
    
```bash
 wsl --list --verbose
```

### Complie the Code

```bash
    cd /mnt/d/GitHub/unparallel-parallelism/src

    nvcc vector_add.cu -o vector_add
```

### Run the Code

```bash
    ./vector_add

    # With profiling with Nsight Systems
    nsys profile ./vector_add
    ncu ./vector_add

```

### SHUTDOWN

> wsl --shutdown

----

## 2. 📦 With Docker container (WIP)

###  Build Image
   
```bash
docker build -t cuda-workshop .

```


### 2.2 Run Image
🐋 Run the container:

```bash
docker run --gpus all -it cuda-workshop

```

----

## 📁 Project Structure


    unparallel-parallelism/
        │
        ├── README.md                               # This README file
        │
        ├── scripts/                                # Setup scripts
        │    └── run_me_first.sh                    # Main setup script to prepare environment
        │    └── cude_temp/cuda_13.0.2_580.95.05_linux.run    # CUDA Toolkit installer (Linux WSL2)
        │
        └── src/                                    # Project source code
             └── vector_add.cu # Sample CUDA program
             └── ...
        


### 📝 Notes

- `scripts/` – contains all setup and helper scripts.  
- `src/` – contains your main project code that you will compile and run inside WSL2/Docker.  


This structure keeps **setup**, **source code**, and **tooling** clearly separated, making the workshop easier to follow.


---

## Refrence
- [CUDA Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#)
- [CUDA Course](https://github.com/Infatoshi/cuda-course)
- [CUDA-Programming](https://github.com/brucefan1983/CUDA-Programming)
- [CUDA Officialdocs CUDA Toolkit 13.0 Update 2](https://developer.nvidia.com/cuda-downloads)
- [CUDA Sample Code](https://github.com/NVIDIA/cuda-samples)
- [NVIDIA/CUDA Docker Images](https://hub.docker.com/r/nvidia/cuda/tags)
- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA)
- [CUDA Lib Sample](https://github.com/NVIDIA/CUDALibrarySamples)