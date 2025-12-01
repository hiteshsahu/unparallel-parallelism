# 🧮 Unparallel Parallelism

Welcome to **Unparallel Parallelism**, a fun workshop exploring the quirks and wonders of parallel computing with CUDA and GPUs.  

Here, we’ll go from “what is a thread?” to “how do I make thousands of them work… or misbehave?”  


### ⚠️ Disclaimer

No GPUs were harmed in the making of this workshop. Any kernel panics are purely educational.


## 📁 Folder Structure


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


-------------------------------------------------------------------------------------------------------------------------------------------------

# ⚙️ **SETUP**

## 💻 Prerequisites
- 🐧 Windows or Linux machine with NVIDIA GPU. Below instruction are meant for Windows machine but you can do the same in Linux machine.
- ⛔ MacOS not supported by CUDA

## ⚡Setup

### 1. Option 1 Using shell Script 📝

One shot approach to setup everything with help of script.

- Install WSL and Linux from step 2.1
- Install NVIDIA Drivers
- Install rest of tools

    ```bash
    wsl
    cd /mnt/d/GitHub/unparallel-parallelism/scripts
     chmod +x run_me_first.sh
    ./run_me_first.sh
    ```

### 2. Option 2 Manual Setup 🔧

Manually setup required tools

### 2.1. 🐧 Install [`WSL2`](https://learn.microsoft.com/en-us/windows/wsl/) and Ubuntu on the Host Machine

- This will give you a Linux environment where Docker + NVIDIA GPU works.

  - 🐧 VERSION 2 → WSL2 (required for GPU support).
  - ⛔ VERSION 1 → WSL1 (no GPU support).

    ```
    wsl --install -d Ubuntu-22.04
    wsl --set-default-version 2
    ```

### 2.2 🎮 NVIDIA GPU drivers for WSL Linux on the Host Machine

-   Download NVIDIA drivers for WSL: [https://developer.nvidia.com/cuda/wsl](https://developer.nvidia.com/cuda/wsl)

    After installation, check:

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

### 2.3. 🐳 Install Docker on WSL2 Linux

- Update your WSL2 Ubuntu

    ```bash
    sudo apt update
    sudo apt upgrade -y
    sudo apt install -y apt-transport-https ca-certificates curl software-properties-common gnupg lsb-release
    ```

- Add Docker’s official repository

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
    docker run hello-world
```

### 2.4 🌀 Enable GPU support inside WSL2

Test GPU passthrough with [**Official CUDA Docker Image**](https://hub.docker.com/r/nvidia/cuda/tags):


> docker run --gpus all nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04 nvidia-smi

## 📟 Install CUDA CLI inside WSL2 Linux Ubuntu 


- First you will need `gcc` in WSL2 to compile CUDA source

    ```bash
    sudo apt install gcc
    gcc --version // Verify
    ```

Follow the instructions to download [Installer of LInux WSL- Ubuntu 2.0 X86_64](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local)

- It will take some time to download source

    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_580.95.05_linux.run
    ```

- lastly run the installer:
- 
  ```bash
  sudo sh cuda_13.0.2_580.95.05_linux.run
  ```

- Add to `$PATH`

   ```bash
    echo $SHELL

    // Output: /bin/bash or /bin/zsh

    //  If it says bin/bash, add the following lines to the ~/.bashrc rile. 
    //  If it says bin/zsh,  add to the ~/.zshrc file.

    echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    ```

Validate Installation to get info about the **Nvidia CUDA compiler **(`nvcc`) & connected GPU **System Management Interface**(`smi`)

```bash
    nvcc --version
    nvidia-smi
```

-------------------------------------------------------------------------------------------------------------------------------------------------

# ▶️ **RUN THE PROJECT**

## 1. Run with WSL locally

### 1.1 Start WSL 2.0
    
```bash
wsl --list --verbose
```

### 1.2 Complie the Code

```bash
nvcc vector_add.cu -o vector_add
```

### 1.3 Run the Code

```bash
./vector_add


// With profiling with Nsight Systems
nsys profile ./vector_add


ncu ./vector_add

```

SHUTDOWN

> wsl --shutdown



## 2. 📦 With Docker container (WIP)


### 2.1 Build Image
   
```bash
docker build -t cuda-workshop .
```


### 2.2 Run Image
🐋 Run the container:

```bash
docker run --gpus all -it cuda-workshop
```

---

## Refrence

- [CUDA Officialdocs CUDA Toolkit 13.0 Update 2](https://developer.nvidia.com/cuda-downloads)
- [CUDA Sample Code](https://github.com/NVIDIA/cuda-samples)
- [NVIDIA/CUDA Docker Images](https://hub.docker.com/r/nvidia/cuda/tags)