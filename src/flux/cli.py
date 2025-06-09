import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
from fire import Fire
from transformers import pipeline

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5, save_image

NSFW_THRESHOLD = 0.85


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
        "- '/w <width>' will set the width of the generated image\n"
        "- '/h <height>' will set the height of the generated image\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/w"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, width = prompt.split()
            options.width = 16 * (int(width) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        elif prompt.startswith("/h"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, height = prompt.split()
            options.height = 16 * (int(height) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        elif prompt.startswith("/g"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, guidance = prompt.split()
            options.guidance = float(guidance)
            print(f"Setting guidance to {options.guidance}")
        elif prompt.startswith("/s"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, seed = prompt.split()
            options.seed = int(seed)
            print(f"Setting seed to {options.seed}")
        elif prompt.startswith("/n"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, steps = prompt.split()
            options.num_steps = int(steps)
            print(f"Setting number of steps to {options.num_steps}")
        elif prompt.startswith("/q"):
            print("Quitting")
            return None
        else:
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    if prompt != "":
        options.prompt = prompt
    return options


@torch.inference_mode()
def main(
    name: str = "flux-dev",
    width: int = 768,
    height: int = 768,
    seed: int | None = None,
    prompt: str = (
        "a photo of a forest with mist swirling around the tree trunks. The word "
        '"FLUX" is painted over it in big, red brush strokes with visible texture'
    ),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    guidance: float = 3.5,
    offload: bool = True,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
        name: Name of the model to load
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
    """
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 24

    # allow for packing and conversion to latent space
    height = 16 * (height // 16)
    width = 16 * (width // 16)

    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    # init all components
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    if loop:
        opts = parse_prompt(opts)

    while opts is not None:
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        t_start = time.perf_counter()

        # prepare input
        #print("开始准备输入...")
        t_prepare_start = time.perf_counter()
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=torch_device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        opts.seed = None
        if offload:
            ae = ae.cpu()
            torch.cuda.empty_cache()
            t5, clip = t5.to(torch_device), clip.to(torch_device)
        inp = prepare(t5, clip, x, prompt=opts.prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))
        t_prepare_end = time.perf_counter()
        #print(f"输入准备完成，耗时: {t_prepare_end - t_prepare_start:.2f}s")

        # offload TEs to CPU, load model to gpu
        #print("加载模型...")
        t_model_load_start = time.perf_counter()
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)
        t_model_load_end = time.perf_counter()
        #print(f"模型加载完成，耗时: {t_model_load_end - t_model_load_start:.2f}s")

        # denoise initial noise
        #print("开始去噪过程...")
        t_denoise_start = time.perf_counter()
        x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)
        t_denoise_end = time.perf_counter()
        #print(f"去噪完成，耗时: {t_denoise_end - t_denoise_start:.2f}s")

        # offload model, load autoencoder to gpu
        #print("加载自编码器...")
        t_ae_load_start = time.perf_counter()
        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)
        t_ae_load_end = time.perf_counter()
        #print(f"自编码器加载完成，耗时: {t_ae_load_end - t_ae_load_start:.2f}s")

        # decode latents to pixel space
        #print("开始解码到像素空间...")
        t_decode_start = time.perf_counter()
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = ae.decode(x)
        t_decode_end = time.perf_counter()
        #print(f"解码完成，耗时: {t_decode_end - t_decode_start:.2f}s")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        fn = output_name.format(idx=idx)
        total_time = t_end - t_start
        print(f"\n=== 时间统计 ===")
        print(f"输入准备: {t_prepare_end - t_prepare_start:.2f}s ({((t_prepare_end - t_prepare_start)/total_time*100):.1f}%)")
        print(f"模型加载: {t_model_load_end - t_model_load_start:.2f}s ({((t_model_load_end - t_model_load_start)/total_time*100):.1f}%)")
        print(f"去噪过程: {t_denoise_end - t_denoise_start:.2f}s ({((t_denoise_end - t_denoise_start)/total_time*100):.1f}%)")
        print(f"自编码器加载: {t_ae_load_end - t_ae_load_start:.2f}s ({((t_ae_load_end - t_ae_load_start)/total_time*100):.1f}%)")
        print(f"解码过程: {t_decode_end - t_decode_start:.2f}s ({((t_decode_end - t_decode_start)/total_time*100):.1f}%)")
        print(f"总耗时: {total_time:.2f}s")
        print(f"保存文件: {fn}")

        idx = save_image(nsfw_classifier, name, output_name, idx, x, add_sampling_metadata, prompt)

        if loop:
            print("-" * 80)
            opts = parse_prompt(opts)
        else:
            opts = None


def app():
    Fire(main)


if __name__ == "__main__":
    app()
