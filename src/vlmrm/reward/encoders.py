from pathlib import Path
from typing import List, Protocol

import open_clip
import torch
import torch.nn as nn
from einops import rearrange, reduce
from torch.amp.autocast_mode import autocast
from vlmrm.contrib.open_clip.transform import VICLIP_MEAN, VICLIP_STD, image_transform
from vlmrm.contrib.viclip import get_viclip


class TextEncoder(Protocol):
    def encode_text(self, x) -> torch.Tensor:
        ...


class VideoEncoder(Protocol):
    def encode_video(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def subsample(self, x: torch.Tensor) -> torch.Tensor:
        ...


class CLIP(nn.Module):
    def __init__(self, model_name: str, pretrained: str, cache_dir: str):
        super().__init__()

        self._model: open_clip.model.CLIP = open_clip.create_model(
            model_name=model_name,
            pretrained=pretrained,
            cache_dir=cache_dir,
        )  # type: ignore
        assert isinstance(self._model, open_clip.model.CLIP)
        size = self._model.visual.image_size
        image_size: int = size if isinstance(size, int) else size[0]  # type: ignore
        self._transform = image_transform(image_size)
        self.expected_n_frames = None

    @torch.inference_mode()
    def encode_text(self, x: List[str]) -> torch.Tensor:
        tokens = open_clip.tokenize(x)
        encoded = self._model.encode_text(tokens).float()
        encoded = encoded / encoded.norm(dim=-1, keepdim=True)
        print(f"{encoded.shape=}")
        return encoded

    @torch.inference_mode()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._model.encode_image(x, normalize=True)
        return encoded

    def subsample(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.inference_mode()
    def encode_video(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[3] != 3:
            x = x.permute(0, 1, 2, 5, 3, 4)
        with torch.no_grad(), autocast("cuda", enabled=torch.cuda.is_available()):
            n_frames, n_windows, n_episodes, *_ = x.shape
            x = rearrange(x, "n_f n_w n_e c h w -> (n_f n_w n_e) c h w")
            # Embed every frame using CLIP
            x = self._transform(x)
            frame_embed = self._model.encode_image(x, normalize=True)
            # Calculate a per-window embedding by averaging all frame embeddings in the window
            window_embed = reduce(
                frame_embed,
                "(n_f n_w n_e) d -> n_w n_e d",
                reduction="mean",
                n_f=n_frames,
                n_w=n_windows,
                n_e=n_episodes,
            )
        return window_embed


class ViCLIP(nn.Module):
    def __init__(self, cache_dir: str, frames_per_video: int = 8) -> None:
        super().__init__()
        model_name = "ViCLIP-L_InternVid-FLT-10M.pth"
        path = Path(cache_dir) / model_name
        self._model, self._tokenizer = get_viclip(
            "l", path.absolute().as_posix(), frames_per_video=frames_per_video
        )
        size = self._model.inputs_image_res
        self._transform = image_transform(size, mean=VICLIP_MEAN, std=VICLIP_STD)
        self.expected_n_frames = self._model.video_input_num_frames

    @torch.inference_mode()
    def encode_text(self, x: List[str]) -> torch.Tensor:
        result = [self._model.get_text_features(t, self._tokenizer) for t in x]
        result = torch.cat(result)
        return result

    def subsample(self, x: torch.Tensor) -> torch.Tensor:
        n_frames, *_ = x.shape
        step = n_frames // self.expected_n_frames
        x = x[::step, ...][: self.expected_n_frames, ...]
        return x

    @torch.inference_mode()
    def encode_video(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[3] != 3:
            x = x.permute(0, 1, 2, 5, 3, 4)
        with torch.no_grad(), autocast("cuda", enabled=torch.cuda.is_available()):
            n_frames, n_windows, n_episodes, *_ = x.shape

            assert n_frames >= self.expected_n_frames

            # Take only n_frames frames, evenly spaced
            x = self.subsample(x)

            x = rearrange(x, "n_f n_w n_e c h w -> (n_f n_w n_e) c h w")
            x = self._transform(x)
            x = rearrange(
                x,
                "(n_f n_w n_e) c h w -> (n_w n_e) n_f c h w",
                n_f=self.expected_n_frames,
                n_w=n_windows,
                n_e=n_episodes,
            )

            # The episodes are the different samples in a batch
            # The window, i.e. the frames, are the one video
            window_embed = self._model.get_vid_features(x)
            window_embed = rearrange(
                window_embed, "(n_w n_e) d -> n_w n_e d", n_w=n_windows, n_e=n_episodes
            )

        return window_embed
