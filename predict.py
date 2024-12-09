from typing import List
import logging

from cog import BasePredictor, Input, Path as CogPath
from runner import HunyanVideoRunner

_log = logging.getLogger("cog_predictor")


class Predictor(BasePredictor):
    def setup(self) -> None:
        logging.basicConfig(
            level=logging.INFO,  # Set the logging level
            format="%(asctime)s-%(levelname)s-%(name)s-%(message)s",
            handlers=[logging.StreamHandler()],  # Log to the console
        )
        self._runner = HunyanVideoRunner()

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt for video generation",
            default="A cat walks on the grass, realistic style.",
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="Ugly",
        ),
        width: int = Input(description="Video width", default=960, ge=128, le=1920),
        height: int = Input(description="Video height", default=544, ge=128, le=1080),
        video_length: int = Input(
            description="Number of frames", default=65, ge=16, le=256
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps", default=40, ge=1, le=100
        ),
        embedded_guidance_scale: float = Input(
            description="Embedded guidance scale", default=6.0, ge=1.0, le=20.0
        ),
        flow_shift: float = Input(
            description="Flow shift parameter", default=7.0, ge=0.0, le=20.0
        ),
        seed: int = Input(description="Random seed", default=None),
    ) -> CogPath:
        """Run video generation inference"""
        video_path = self._runner.run_hunyan_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=video_length,
            seed=seed,
            embedded_guidance_scale=embedded_guidance_scale,
            steps=num_inference_steps,
            flow_shift=flow_shift,
        )
        return CogPath(video_path)
