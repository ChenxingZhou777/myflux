from typing import Optional, TYPE_CHECKING
import time

import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from platform import system
from loguru import logger

if TYPE_CHECKING:
    from flux_pipeline import FluxPipeline

if system() == "Windows":
    MAX_RAND = 2**16 - 1
else:
    MAX_RAND = 2**32 - 1


class AppState:
    model: "FluxPipeline"


class FastAPIApp(FastAPI):
    state: AppState


class GenerateArgs(BaseModel):
    prompt: str
    width: Optional[int] = Field(default=720)
    height: Optional[int] = Field(default=1024)
    num_steps: Optional[int] = Field(default=24)
    guidance: Optional[float] = Field(default=3.5)
    seed: Optional[int] = Field(
        default_factory=lambda: np.random.randint(0, MAX_RAND), gt=0, lt=MAX_RAND
    )
    strength: Optional[float] = 1.0
    init_image: Optional[str] = None


app = FastAPIApp()


@app.post("/generate")
def generate(args: GenerateArgs):
    """
    Generates an image from the Flux flow transformer.

    Args:
        args (GenerateArgs): Arguments for image generation:

            - `prompt`: The prompt used for image generation.

            - `width`: The width of the image.

            - `height`: The height of the image.

            - `num_steps`: The number of steps for the image generation.

            - `guidance`: The guidance for image generation, represents the
                influence of the prompt on the image generation.

            - `seed`: The seed for the image generation.

            - `strength`: strength for image generation, 0.0 - 1.0.
                Represents the percent of diffusion steps to run,
                setting the init_image as the noised latent at the
                given number of steps.

            - `init_image`: Base64 encoded image or path to image to use as the init image.

    Returns:
        StreamingResponse: The generated image as streaming jpeg bytes.
    """
    api_start_time = time.time()
    
    logger.info(f"🌐 API Request received:")
    logger.info(f"   📝 Prompt: {args.prompt}")
    logger.info(f"   📐 Size: {args.width}x{args.height}")
    logger.info(f"   🔄 Steps: {args.num_steps}")
    logger.info(f"   🔢 Seed: {args.seed}")
    
    result = app.state.model.generate(**args.model_dump())
    
    api_total_time = time.time() - api_start_time
    logger.info(f"🌐 API Response ready in {api_total_time:.2f}s")
    
    return StreamingResponse(result, media_type="image/jpeg")
