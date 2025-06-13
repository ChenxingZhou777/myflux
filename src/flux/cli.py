import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from glob import iglob
from typing import List

import torch
from fire import Fire
from transformers import pipeline

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5, save_image

NSFW_THRESHOLD = 0.85


def get_gpu_memory_info():
    """获取GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return {
            'allocated': allocated,
            'reserved': reserved, 
            'total': total_memory,
            'free': total_memory - reserved
        }
    return None


def print_gpu_memory(stage_name):
    """打印当前阶段的GPU显存使用情况"""
    memory_info = get_gpu_memory_info()
    if memory_info:
        print(f"🔥 [{stage_name}] GPU显存: 已分配 {memory_info['allocated']:.2f}GB | "
              f"已保留 {memory_info['reserved']:.2f}GB | "
              f"空闲 {memory_info['free']:.2f}GB | "
              f"总计 {memory_info['total']:.2f}GB")


def optimized_denoise_with_loading(
    model,
    img,
    img_ids,
    txt,
    txt_ids,
    vec,
    timesteps: List[float],
    guidance: float = 4.0,
    torch_device: torch.device = None,
):
    """优化的去噪函数 - 通过重新安排执行顺序来减少等待时间"""
    
    print("🚀 开始优化推理...")
    
    # 记录开始时间
    total_start = time.time()
    
    # 第一步：立即开始模型加载（在后台）
    print("📦 开始加载模型到GPU...")
    load_start = time.time()
    
    # 同步加载，避免多线程问题
    model = model.to(torch_device)
    
    load_end = time.time()
    print(f"✅ 模型加载完成，耗时: {load_end - load_start:.2f}s")
    
    # 确保所有输入数据都在正确的设备上
    print("🔧 确保数据设备一致性...")
    img = img.to(torch_device)
    img_ids = img_ids.to(torch_device)
    txt = txt.to(torch_device)
    txt_ids = txt_ids.to(torch_device)
    vec = vec.to(torch_device)
    
    # 执行标准去噪流程
    print("🚀 开始推理...")
    denoise_start = time.time()
    
    guidance_vec = torch.full((img.shape[0],), guidance, device=torch_device, dtype=img.dtype)
    
    for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        step_start = time.time()
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=torch_device)
        
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        
        img = img + (t_prev - t_curr) * pred
        
        step_time = time.time() - step_start
        if step_idx == 0:
            print(f"🚀 首个推理步骤耗时: {step_time:.3f}s")
        elif step_idx % max(1, (len(timesteps) - 1) // 4) == 0:
            print(f"🔄 Step {step_idx}/{len(timesteps)-1}, 耗时: {step_time:.3f}s")
    
    denoise_end = time.time()
    total_end = time.time()
    
    print(f"✅ 优化推理完成！")
    print(f"   模型加载: {load_end - load_start:.2f}s")
    print(f"   推理计算: {denoise_end - denoise_start:.2f}s") 
    print(f"   总耗时: {total_end - total_start:.2f}s")
    
    return img


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
    overlap_decoding: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    guidance: float = 3.5,
    offload: bool = True,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    fast_memory: bool = True,
    pipeline_loading: bool = True,
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
        fast_memory: Use optimized memory management (del model instead of model.cpu())
        pipeline_loading: Use pipeline loading optimization (load model layers while denoising)
        overlap_decoding: Use overlap optimization for model.cpu(), ae.decoder.to() and decode operations
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
    print_gpu_memory("初始化前")
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)
    print_gpu_memory("模型初始化完成")

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
        print_gpu_memory("准备输入前")
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
        # if offload:
        #     #ae = ae.cpu()
        #     torch.cuda.empty_cache()
        #     t5, clip = t5.to(torch_device), clip.to(torch_device)
        inp = prepare(t5, clip, x, prompt=opts.prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))
        t_prepare_end = time.perf_counter()
        print_gpu_memory("输入准备完成")

        # offload TEs to CPU, load model to gpu
        t_model_load_start = time.perf_counter()
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            print_gpu_memory("清空缓存后")
            
            # 使用优化加载
            if pipeline_loading:
                print("🚀 启用优化加载...")
                print_gpu_memory("优化推理前")
                t_denoise_start = time.perf_counter()
                x = optimized_denoise_with_loading(
                    model=model,
                    img=inp["img"],
                    img_ids=inp["img_ids"], 
                    txt=inp["txt"],
                    txt_ids=inp["txt_ids"],
                    vec=inp["vec"],
                    timesteps=timesteps,
                    guidance=opts.guidance,
                    torch_device=torch_device
                )
                t_denoise_end = time.perf_counter()
                t_model_load_end = t_denoise_end  
                print_gpu_memory("优化推理完成")
            else:
                model = model.to(torch_device)
                t_model_load_end = time.perf_counter()
                print_gpu_memory("主模型加载完成")
                
                # denoise initial noise
                print_gpu_memory("去噪前")
                t_denoise_start = time.perf_counter()
                x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)
                t_denoise_end = time.perf_counter()
                print_gpu_memory("去噪完成")
        else:
            model = model.to(torch_device)
            t_model_load_end = time.perf_counter()
            print_gpu_memory("主模型加载完成")
            
            # denoise initial noise
            print_gpu_memory("去噪前")
            t_denoise_start = time.perf_counter()
            x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)
            t_denoise_end = time.perf_counter()
            print_gpu_memory("去噪完成")

        # offload model, load autoencoder to gpu
        t_ae_load_start = time.perf_counter()
        
        if offload and overlap_decoding:
            # 🚀 使用Overlap优化
            print_gpu_memory("Overlap解码优化前")
            x, overlap_timing = overlap_decoding_optimization(
                model=model,
                ae=ae,
                x=x,
                opts=opts,
                torch_device=torch_device,
                offload=offload,
                loop=loop,
                fast_memory=fast_memory,
            )
            # 对于overlap模式，我们需要调整计时
            t_ae_load_end = time.perf_counter()
            t_decode_start = overlap_timing['decode']['start'] if 'decode' in overlap_timing else t_ae_load_start
            t_decode_end = overlap_timing['decode']['end'] if 'decode' in overlap_timing else t_ae_load_end
            print_gpu_memory("Overlap解码优化完成")
            
        elif offload:
            # 🔄 传统模式
            if pipeline_loading and not loop:
                t_model_del_start = time.perf_counter()
                del model
                t_model_del_end = time.perf_counter()
            else:
                # 交互模式或传统模式：保留模型以便重用
                reason = "交互模式" if loop else "传统模式"
                print(f"🔄 {reason}：将模型移动到CPU...")
                t_model_cpu_start = time.perf_counter()
                model.cpu()
                t_model_cpu_end = time.perf_counter()
                print(f"model.cpu() 耗时: {t_model_cpu_end - t_model_cpu_start:.4f}s")
            
            # 计时 torch.cuda.empty_cache()
            # t_empty_cache_start = time.perf_counter()
            torch.cuda.empty_cache()
            #t_empty_cache_end = time.perf_counter()
            # print(f"torch.cuda.empty_cache() 耗时: {t_empty_cache_end - t_empty_cache_start:.4f}s")
            
            # 计时 ae.decoder.to(x.device)
            t_ae_to_device_start = time.perf_counter()
            ae.decoder.to(x.device)
            t_ae_to_device_end = time.perf_counter()
            print(f"ae.decoder.to(x.device) 耗时: {t_ae_to_device_end - t_ae_to_device_start:.4f}s")
            
            t_ae_load_end = time.perf_counter()
            print_gpu_memory("自编码器加载完成")
            print(f"自编码器加载完成，总耗时: {t_ae_load_end - t_ae_load_start:.2f}s")

            # decode latents to pixel space
            print_gpu_memory("解码前")
            t_decode_start = time.perf_counter()
            x = unpack(x.float(), opts.height, opts.width)
            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                x = ae.decode(x)
            t_decode_end = time.perf_counter()
            print_gpu_memory("解码完成")
            
        else:
            # 非offload模式，直接解码
            t_ae_load_end = time.perf_counter()
            print_gpu_memory("解码前")
            t_decode_start = time.perf_counter()
            x = unpack(x.float(), opts.height, opts.width)
            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                x = ae.decode(x)
            t_decode_end = time.perf_counter()
            print_gpu_memory("解码完成")

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
        print_gpu_memory("推理完成")

        if loop:
            print("-" * 80)
            # 在loop模式下，如果使用了fast_memory或overlap_decoding，需要重新加载模型
            if offload and (fast_memory or overlap_decoding):
                print("🔄 重新加载模型用于下次推理...")
                model = load_flow_model(name, device="cpu")
            opts = parse_prompt(opts)
        else:
            opts = None


def overlap_decoding_optimization(
    model,
    ae,
    x,
    opts,
    torch_device,
    offload: bool = True,
    loop: bool = False,
    fast_memory: bool = False,
):
    """
    Overlap优化：并发执行model.cpu()、ae.decoder.to()和解码过程
    
    Args:
        model: Flux模型
        ae: 自编码器
        x: 去噪后的潜在表示
        opts: 采样选项
        torch_device: torch设备
        offload: 是否启用offload
        loop: 是否为循环模式
        fast_memory: 是否使用快速内存模式
    
    Returns:
        decoded_x: 解码后的图像张量
    """
    
    print("🚀 启用Overlap解码优化...")
    
    # 计时开始
    t_overlap_start = time.perf_counter()
    
    # 准备同步对象
    ae_ready_event = threading.Event()
    model_cpu_done_event = threading.Event()
    decode_result = {}  # 用于存储解码结果
    timing_results = {}  # 用于存储计时结果
    
    def model_cpu_task():
        """模型转CPU的任务"""
        nonlocal model  # 允许修改外部变量
        print("🔄 [线程1] 开始model.cpu()...")
        t_model_cpu_start = time.perf_counter()
        
        if fast_memory and not loop:
            # 快速内存模式：直接删除模型
            del model
            model = None  # 设置为None，避免后续访问错误
            operation_type = "del model"
        else:
            # 传统模式：移动到CPU
            model.cpu()
            operation_type = "model.cpu()"
        
        torch.cuda.empty_cache()
        t_model_cpu_end = time.perf_counter()
        
        timing_results['model_cpu'] = {
            'start': t_model_cpu_start,
            'end': t_model_cpu_end,
            'operation': operation_type
        }
        
        print(f"✅ [线程1] {operation_type} 完成，耗时: {t_model_cpu_end - t_model_cpu_start:.4f}s")
        model_cpu_done_event.set()
    
    def ae_loading_task():
        """自编码器加载的任务"""
        print("🔄 [线程2] 开始ae.decoder.to(device)...")
        t_ae_to_device_start = time.perf_counter()
        
        ae.decoder.to(x.device)
        
        t_ae_to_device_end = time.perf_counter()
        timing_results['ae_loading'] = {
            'start': t_ae_to_device_start,
            'end': t_ae_to_device_end
        }
        
        print(f"✅ [线程2] ae.decoder.to(device) 完成，耗时: {t_ae_to_device_end - t_ae_to_device_start:.4f}s")
        ae_ready_event.set()
    
    def decode_task():
        """解码任务 - 需要等待ae准备完成"""
        print("⏳ [线程3] 等待ae.decoder加载完成...")
        ae_ready_event.wait()  # 等待AE加载完成
        
        print("🔄 [线程3] 开始解码...")
        t_decode_start = time.perf_counter()
        
        # 解码潜在表示到像素空间
        x_unpacked = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            decoded_result = ae.decode(x_unpacked)
        
        t_decode_end = time.perf_counter()
        timing_results['decode'] = {
            'start': t_decode_start,
            'end': t_decode_end
        }
        
        decode_result['x'] = decoded_result
        print(f"✅ [线程3] 解码完成，耗时: {t_decode_end - t_decode_start:.4f}s")
    
    # 使用ThreadPoolExecutor并发执行任务
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 提交所有任务
        future_model_cpu = executor.submit(model_cpu_task)
        future_ae_loading = executor.submit(ae_loading_task)
        future_decode = executor.submit(decode_task)
        
        # 等待所有任务完成
        futures = [future_model_cpu, future_ae_loading, future_decode]
        for future in as_completed(futures):
            try:
                future.result()  # 获取结果，如果有异常会在这里抛出
            except Exception as e:
                print(f"❌ Overlap任务执行失败: {e}")
                raise
    
    t_overlap_end = time.perf_counter()
    
    # 打印性能统计
    print(f"\n🚀 === Overlap优化性能统计 ===")
    total_overlap_time = t_overlap_end - t_overlap_start
    
    if 'model_cpu' in timing_results:
        model_cpu_time = timing_results['model_cpu']['end'] - timing_results['model_cpu']['start']
        print(f"  {timing_results['model_cpu']['operation']}: {model_cpu_time:.4f}s")
    
    if 'ae_loading' in timing_results:
        ae_loading_time = timing_results['ae_loading']['end'] - timing_results['ae_loading']['start']
        print(f"  ae.decoder.to(device): {ae_loading_time:.4f}s")
    
    if 'decode' in timing_results:
        decode_time = timing_results['decode']['end'] - timing_results['decode']['start']
        print(f"  解码过程: {decode_time:.4f}s")
    
    print(f"  Overlap总耗时: {total_overlap_time:.4f}s")
    
    # 计算如果串行执行的时间
    if all(key in timing_results for key in ['model_cpu', 'ae_loading', 'decode']):
        sequential_time = (timing_results['model_cpu']['end'] - timing_results['model_cpu']['start'] +
                          timing_results['ae_loading']['end'] - timing_results['ae_loading']['start'] +
                          timing_results['decode']['end'] - timing_results['decode']['start'])
        speedup = sequential_time / total_overlap_time
        print(f"  估算串行时间: {sequential_time:.4f}s")
        print(f"  🚀 加速比: {speedup:.2f}x")
    
    return decode_result['x'], timing_results


def app():
    Fire(main)

# 🚀 模型内存优化建议:
#
# 使用方法:
#   python -m flux.cli --fast_memory=true   # 默认启用快速内存模式
#   python -m flux.cli --fast_memory=false  # 使用传统模式
#   python -m flux.cli --loop               # 交互模式自动使用传统模式
#   python -m flux.cli --pipeline_loading=true  # 启用流水线加载优化（推荐）
#   python -m flux.cli --pipeline_loading=false # 禁用流水线优化
#   python -m flux.cli --overlap_decoding=true  # 启用Overlap解码优化
#   python -m flux.cli --overlap_decoding=false # 禁用Overlap优化

if __name__ == "__main__":
    app()
