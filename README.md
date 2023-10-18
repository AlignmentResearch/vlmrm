# `vlmrm`

This is the repository from the paper "Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning". We provide training scripts that can be used to reproduce our experiments, and a Python package that can be installed with pip and imported from another project.

Instead of manually specifying reward functions or relying on extensive human feedback to train your reinforcement learning agents, you can now use `vlmrm` to specify tasks from only natural language prompts by leveraging pretrained vision-language models (VLMs) as zero-shot reward models (RMs).

We provide implementations for:

- Utilizing any CLIP model available from the `open_clip` package as the reward model,
- Rendering MuJoCo environments on the GPU using EGL,
- Parallelizing rendering and reward computation across multiple GPUs,
- An adapted version of the SAC and DQN implemented in `stable_baselines3` that allows computing rewards at the end of episodes to increase fps by leveraging GPU batching,
- A working Dockerfile to use with docker + CUDA backend or containerd + kubernetes.

## Citation

You can cite our work by using the following BibTex file:

```bibtex
@article{rocamonde_2023_vision,
  title={{Vision-Language} {Models} are {Zero-Shot} {Reward} {Models} for {Reinforcement} {Learning}},
  author={Rocamonde, Juan and Montesinos, Victoriano and Nava, Elvis and Perez, Ethan and Lindner, David},
  journal={arXiv preprint arXiv:},
  year={2023}
}
```

## Installation

This assumes you are using Python 3.9.

### Development

`pip install -e ".[dev]"` for development.

### Docker

1. Ensure you have docker installed. If not, follow the instructions [here](https://docs.docker.com/engine/install/ubuntu/):

```bash
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install --yes \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin
```

2. Ensure your host machine building the image has the nvidia drivers installed. The simplest way to do this is by running:

```bash
sudo apt install --yes ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
```

3. Next, install the NVIDIA container toolkit:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit nvidia-docker2
sudo systemctl restart docker
```

Now set some variables to be used throughout this README. Replace `<your_username>` with your desired username for your Docker images (e.g. your Hub's username). Insert your wandb API key.

```bash
export DOCKER_USER=<your_username>
export WANDB_API_KEY=<your_api_key>
```

Finally, build the docker image:

```bash
docker build -t $DOCKER_USER/vlmrm:latest . -f docker/Dockerfile
```

Test that everything is running smoothly:

```bash
docker run -it --rm --gpus=all --runtime=nvidia \
    $DOCKER_USER/vlmrm:latest \
    python3 /root/vlmrm/test_fps.py
```

Remember to push the image to a container registry if you want to use it on a cluster.

```bash
sudo docker push $DOCKER_USER/vlmrm:latest
```

The Dockerfile uses the CUDA container image as a base. The CUDA container base is subject to the license found on `docker/NGC-DL-CONTAINER-LICENSE`.

## Usage

### Using Docker

_Note_: if you're using multiple GPUs for rendering and reward computation, NCCL will use shared memory by default to pass tensors across GPUs. However, Docker has a very low limit on shared memory by default (64MB). You can customize this by setting `--shm-size` flag to multiple GBs according to your RAM specs, or disable shared memory entirely and use network (i.e. InfiniBand or IP sockets) to communicate between the CPU sockets by setting `NCCL_SHM_DISABLE=1` as an environment variable.

```bash
docker run -it --rm \
    -v $(pwd):/root/vlmrm -v ~/.cache/models/:/root/.cache/ \
    --gpus=all --runtime=nvidia \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    $DOCKER_USER/vlmrm:latest \
    vlmrm train "$(cat config.yaml)"
```

where `config.yaml` is a YAML file with the following structure:

```yaml
env_name: Humanoid-v4 # RL environment name
base_path: /data/runs/training # Base path to save logs and checkpoints
seed: 42 # Seed for reproducibility
description: Humanoid training using CLIP reward
tags: # Wandb tags
  - training
  - humanoid
  - CLIP
reward:
  name: clip
  pretrained_model: ViT-g-14/laion2b_s34b_b88k # CLIP model name
  # CLIP batch size per synchronous inference step.
  # Batch size must be divisible by n_workers (GPU count)
  # so that it can be shared among workers, and must be a divisor
  # of n_envs * episode_length so that all batches can be of the
  # same size (no support for variable batch size as of now.)
  batch_size: 1600
  alpha: 0.5 # Alpha value of Baseline CLIP (CO-RELATE)
  target_prompts: # Description of the goal state
    - a humanoid robot kneeling
  baseline_prompts: # Description of the environment
    - a humanoid robot
  # Path to pre-saved model weights. When executing multiple runs,
  # mount a volume to this path to avoid downloading the model
  # weights multiple times.
  cache_dir: /root/.cache
rl:
  policy_name: MlpPolicy
  n_steps: 100000 # Total number of simulation steps to be collected.
  n_envs_per_worker: 2 # Number of environments per worker (GPU)
  episode_length: 200 # Desired episode length
  learning_starts: 100 # Number of env steps to collect before training
  train_freq: 200 # Number of collected env steps between training iterations
  batch_size: 64 # SAC buffer sample size per gradient step
  gradient_steps: 1 # Number of samples to collect from the buffer per training step
  tau: 0.005 # SAC target network update rate
  gamma: 0.99 # SAC discount factor
  learning_rate: 3e-4 # SAC optimizer learning rate
logging:
  checkpoint_freq: 800 # Number of env steps between checkpoints
  video_freq: 800 # Number of env steps between videos
  tensorboard_freq: 800 # Number of env steps between tensorboard logs
```

### Using Kubernetes

Example job can be found on the `kube/` folder.

### Using your host machine

Simply run the `vlmrm` command.

```bash
vlmrm train "$(cat config.yaml)"
```

Alternatively, run

```bash
vlmrm train "$(cat EOF
...
EOF
)"
```

replacing `...` with the YAML content.

You can run a job in the background by wrapping the command with `nohup` at the beginning and and `&` at the end. You can also redirect error and info logs to a `log.txt` and save the PID in a file to easily terminate it later if needed:

```bash
nohup vlmrm (...) > log.txt 2>&1 & echo $! > pid.txt && tail -f log.txt
```

### Other tricks

The code uses nested subprocesses and sometimes processes can exist not gracefully and leave defunct processes running. If this occurs during development, you may consider running the following: (USE AT YOUR OWN RISK)

```bash
# Get a list of all zombie processes
zombies=$(ps aux | awk '{ if ($8 == "Z") { print $2 } }')

# For each zombie process, attempt to kill its parent
for pid in $zombies; do
    # Find parent of the zombie
    ppid=$(ps -o ppid= -p $pid)
    echo "Killing parent process $ppid of zombie $pid"
    kill -9 $ppid
done
```
