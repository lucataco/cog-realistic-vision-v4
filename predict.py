# Prediction interface for Cog ⚙️
from cog import BasePredictor, Input, Path
import os
import torch
import math
from diffusers import StableDiffusionPipeline
import tempfile

MODEL_NAME = "SG161222/Realistic_Vision_V4.0"
MODEL_CACHE = "cache"

class Predictor(BasePredictor):
    def base(self, x):
        return int(8 * math.floor(int(x)/8))

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE
        )
        self.pipe.to("cuda")

    def predict(
        self,
        prompt: str = "RAW photo, a portrait photo of a latina woman in casual clothes, natural skin, 8k uhd, high quality, film grain, Fujifilm XT3",
        negative_prompt: str = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",
        steps: int = Input(description=" num_inference_steps", ge=0, le=100, default=20),
        guidance: float = Input(description="Guidance scale (3.5 - 7)", default=5),
        width: int = Input(description="Width", ge=0, le=1920, default=512),
        height: int = Input(description="Height", ge=0, le=1920, default=728),
        seed: int = Input(description="Seed (0 = random, maximum: 2147483647)", default=0),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed == 0:
            seed = int.from_bytes(os.urandom(2), byteorder='big')
        generator = torch.Generator('cuda').manual_seed(seed)

        width = self.base(width)
        height = self.base(height)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator
        ).images[0]

        output_path = Path(tempfile.mkdtemp()) / "ai.png"
        image.save(output_path)

        return  Path(output_path)

