# Quick Start

Add the following packages into your virtual environment:

```
pip install imageio av seaborn
```

## Models

If you want to test S3D, download the required files into your cache directory (default: `.cache`). 

```sh
cd .cache
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy
```

If you want to test ViCLIP, you should put `ViCLIP-L_InternVid-FLT-10M.pth` to the cache directory, too. You can get the file [here](https://huggingface.co/OpenGVLab/ViCLIP/tree/main).

If you want to test GPT-4V, you should put your API key into an .env file in the project-root:

```
# .env
OPENAI_API_KEY=
```

## Data

In the project root, create a folder `data/evaluation` and put your videos in there. In the same directory, place a `data.csv` file contaning four columns: 

- `path`, path to an mp4 file with a video clip, e.g. `data/evaluation/straight_middle.mp4`
- `baseline`, a baseline label for this video (used in projection rewards)
- `label`, a target label for this video (used in all rewards)
- `group`, a group identifier, can be any string (used in the matrix plotting code; different groups are separated by a blank space)

For example, data.csv might look like:

```csv
data/evaluation/intersection_forward_t->b.mp4,"topdown view of a cartoon red car standing in place","topdown view of a cartoon red car at an intersection, going forward",4.1
data/evaluation/intersection_left_t->r.mp4,"topdown view of a cartoon red car standing in place","topdown view of a cartoon red car at an intersection, turning left",4.2
```

## Running the evaluator

Once your models and data are in place, you can set up the evaluation experinemt in the `run_evals.sh` script. By default, this is what the script looks like:

```bash
models=("clip" "viclip" "s3d" "gpt4")

for model in "${models[@]}"; do
    python src/evaluation/evaluator.py -t data/evaluation/data.csv -m "$model" -r logit,projection -a 0.0,0.25,0.50,0.75,1.0 -n 32 -e standardized_improved --standardize
done
```

This will run the evaluator for each model in `models`. The evaluator will then load the given model (`-m`) and run a set of experiments on it, one experiment for each combination of hyperparameters. Each experiment will generate its own confusion matrix, saved by default in `out`. The hyperparameters are:
    - method to compute reward (`-r`), used for non-gpt models
    - alpha to use with the projection reward (`a`)
    - number of frames to average over for the clip model (`-n`)
    - whether to standardize the rewards or not (`--standardize`)
    - the experiment name that is prefixed to all the generated matrices (`-e`)

One you edit `run_evals.sh` with the hyperparameters you want to use, you can run it like this:

```shell
sbatch run_evals.sh
```

## Architecture

This is only a high-level overview, ask Evžen for details. Do not spend too much time trying to figure out things on your own :-)

The evaluator loads a model. All the models (but GPT-4V) have a common interface called `Encoder`. Encoders can encode both text and videos and are specified in `src/vlmrm/reward/encoders.py`. If you want to add another model to eval, implementing the Encoder interface for it would be the place to start.

Encoders expect the video to come pre-segmented into pieces called windows. Windows can overlap, but have all the same length, which we call `n_frames` below. In addition, because the class is used by vlmrm, the videos are expected to come from multiple episodes. The complete type of `encode_videos` is then:

```haskell
encode_video :: (n_frames n_windows n_episodes c h w) -> (n_windows n_episodes embed_dim)
```

Once we have the encodings for each window, we can pass them to a `Reward` function — this is again an interface which you'd need to implement if you wanted to support additional reward functions. It is specified in `src/vlmrm/reward/rewards.py`. A reward is a stateful function that computes a reward for every window it is passed; i.e. its type is:

```
reward :: (n_windows n_episodes embed_dim) -> (n_windows n_episodes)
```

I said _stateful_ function since, by default, the reward needs to be initialized first with the target label wrt which the reward should be computed. This is again caused by this interface being used in vlmrm. Feel free to implement your own reward function that get the target label on each call (or see `projection_reward` and `logit_reward` in `src/vlmrm/reward/rewards.py`).
