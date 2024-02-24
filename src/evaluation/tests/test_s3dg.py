import pytest
import torch
from modeling.s3dg import S3D


@pytest.fixture
def net():
    embedding_dim = 512
    net = S3D(".cache/s3d_dict.npy", embedding_dim)
    # Load the model weights
    net.load_state_dict(torch.load(".cache/s3d_howto100m.pth"))
    # Evaluation mode
    net = net.eval()
    return net


def test_original_s3dg(net):
    # Instantiate the model

    # Video input should be of size Batch x 3 x T x H x W and normalized to [0, 1]
    video = torch.rand(2, 3, 32, 224, 224)

    # Video inference
    video_output = net(video)
    for key, value in video_output.items():
        print(key, value.shape)

    # Text inference
    text_output = net.text_module(["open door", "cut tomato"])
    for key, value in text_output.items():
        print(key, value.shape)
