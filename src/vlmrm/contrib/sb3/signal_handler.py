import logging
import sys
from typing import Optional

from stable_baselines3.common.base_class import BaseAlgorithm

from vlmrm.contrib.sb3.save_model import save_model, save_replay_buffer

logger = logging.getLogger(__name__)

checkpoint_dir: Optional[str] = None
model: Optional[BaseAlgorithm] = None


def end_signal_handler(signal, frame):
    if checkpoint_dir is not None and model is not None:
        logger.info("Received end signal. Cleaning up...")
        save_model(checkpoint_dir, model)
        save_replay_buffer(checkpoint_dir, model)
    sys.exit(0)
