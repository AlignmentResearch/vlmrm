import pytest
from typing import List, Callable

import torch
import numpy as np

from modeling.util import load_prompts, load_video, get_video_batch, strip_directories_and_extension, make_heatmap


@pytest.fixture
def path_to_prompts(tmp_path):
    #content of some_prompts.txt
    SOME_PROMPTS_TXT_CONTENT = """A stick model of a dog running
A stick model of a dog running on back legs
A stick model of a dog running in small steps
A stick model of a dog falling and crawling
"""

    d = tmp_path / "prompts"
    d.mkdir()
    p = d / "some_prompts.txt"
    p.write_text(SOME_PROMPTS_TXT_CONTENT)
    return p

@pytest.fixture
def trajectories_path(tmp_path):
    d = tmp_path / "trajectories_lists"
    d.mkdir()
    p = d / "cockatoos.txt"
    p.write_text("""imageio:cockatoo.mp4
imageio:cockatoo.mp4
imageio:cockatoo.mp4
""")
    return p

@pytest.fixture
def dummy_prepare_video():
    def fn(video: np.ndarray, n_frames: int, verbose: bool):
        if verbose:
            print("video.shape", video.shape)
        return torch.from_numpy(video[:n_frames]).permute(3, 0, 1, 2).unsqueeze(0)

    return fn

@pytest.mark.parametrize(["verbose"],[[True], [False]])
def test_load_promts(path_to_prompts: str, verbose: bool):
    prompts = load_prompts(path_to_prompts, verbose)
    assert isinstance(prompts, List)
    assert all(isinstance(p, str) for p in prompts)

def test_load_video(path_to_video: str = "imageio:cockatoo.mp4"):
    video = load_video(path_to_video)
    assert isinstance(video, np.ndarray)

@pytest.mark.parametrize(["n_frames"], [[8], [32]])
@pytest.mark.parametrize(["verbose"], [[True], [False]])
def test_get_video_batch(trajectories_path: str, dummy_prepare_video: Callable, n_frames: int, verbose: bool):
    batch, video_paths = get_video_batch(trajectories_path, dummy_prepare_video, n_frames, verbose)
    assert isinstance(batch, torch.Tensor)
    assert len(video_paths) == batch.shape[0]

@pytest.mark.parametrize(["path", "expected_answer"], [
    ["~/trajectories/cheetah/cheetah_back_legs_0.mp4", "cheetah_back_legs_0"],
    ["/data/username/trajectories/cheetah/cheetah_back_legs_0.mp4", "cheetah_back_legs_0"],
    ["trajectory.mp4", "trajectory"],
    ["/data/username/unexpected.filename.with.dots.mp4", "unexpected"]
])
def test_strip_directories_and_extension(path: str, expected_answer: str):
    assert strip_directories_and_extension(path) == expected_answer

@pytest.mark.parametrize(["similarity_matrix"], [
    [np.ones((2,2)) / 2],
    [np.zeros((2,2))]
])
def test_make_heatmap(tmp_path, similarity_matrix: np.ndarray, trajectories_names=["first", "second"], labels=["good", "bad"], experiment_id="check"):
    make_heatmap(similarity_matrix, trajectories_names, labels, tmp_path, experiment_id)
    assert (tmp_path / f"{experiment_id}.png").exists()