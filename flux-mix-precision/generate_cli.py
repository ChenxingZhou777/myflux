#!/usr/bin/env python3
"""
Flux FP8 命令行直接生图工具

使用示例:
python generate_cli.py --config configs/config-dev-offload-1-4090.json \
                       --prompt "a beautiful landscape" \
                       --output my_image.jpg \
                       --width 1024 --height 1024 \
                       --steps 24 --guidance 3.5 --seed 12345

交互模式:
python generate_cli.py --config configs/config-dev-offload-1-4090.json --loop
"""

import argparse
import time
from pathlib import Path
from flux_pipeline import FluxPipeline
from loguru import logger


def generate_single_image(pipeline, prompt, width, height, steps, guidance, seed, quality, output, silent, image_counter=None):
    """生成单张图片的函数"""
    gen_start = time.time()
    result, actual_seed = pipeline.generate(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=steps,
        guidance=guidance,
        seed=seed,
        silent=silent,
        return_seed=True,
        jpeg_quality=quality
    )
    gen_time = time.time() - gen_start
    
    # 处理输出文件名
    if image_counter is not None:
        # 循环模式：使用递增的计数器
        if "{}" in output:
            output_path = Path(output.format(image_counter))
        else:
            # 如果用户没有使用模板格式，则添加计数器后缀
            output_path = Path(output)
            stem = output_path.stem
            suffix = output_path.suffix
            output_path = output_path.parent / f"{stem}_{image_counter:03d}{suffix}"
    else:
        # 单次模式：查找下一个可用的文件名
        if "{}" in output:
            counter = 1
            while True:
                output_path = Path(output.format(counter))
                if not output_path.exists():
                    break
                counter += 1
        else:
            output_path = Path(output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(result.getvalue())
    
    return output_path, actual_seed, gen_time


def main():
    parser = argparse.ArgumentParser(description="Flux命令行生图工具")
    
    # 必需参数
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        default="configs/config-dev-offload-1-4090.json",
        help="配置文件路径 (默认: configs/config-dev-offload-1-4090.json)"
    )
    parser.add_argument(
        "--prompt", "-p", 
        type=str, 
        required=False,
        help="生图提示词（循环模式下可选）"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="output/img_{}.jpg",
        help="输出图片路径模板 (默认: output/img_{}.jpg，{}会被替换为递增数字)"
    )
    
    # 可选参数
    parser.add_argument("--width", "-w", type=int, default=1024, help="图片宽度 (默认: 1024)")
    parser.add_argument("--height", "-H", type=int, default=1024, help="图片高度 (默认: 1024)")
    parser.add_argument("--steps", "-s", type=int, default=24, help="推理步数 (默认: 24)")
    parser.add_argument("--guidance", "-g", type=float, default=3.5, help="引导强度 (默认: 3.5)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (默认: 随机)")
    parser.add_argument("--quality", "-q", type=int, default=95, help="JPEG质量 (默认: 95)")
    parser.add_argument("--silent", action="store_true", help="静默模式，不显示详细信息")
    parser.add_argument("--loop", "-l", action="store_true", help="交互循环模式，加载模型后可多次生图")
    
    args = parser.parse_args()
    
    # 如果是循环模式但没有提供prompt，这是正常的
    if not args.loop and not args.prompt:
        logger.error("❌ 非循环模式下必须提供 --prompt 参数")
        return 1
    
    # 检查配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"❌ 配置文件不存在: {args.config}")
        return 1

    try:
        # 加载模型
        load_start = time.time()
        if not args.silent:
            logger.info("🔄 正在加载模型...")
        
        pipeline = FluxPipeline.load_pipeline_from_config_path(
            str(config_path),
            debug=not args.silent
        )
        
        load_time = time.time() - load_start
        if not args.silent:
            logger.info(f"✅ 模型加载完成，耗时 {load_time:.1f}s")
        
        if args.loop:
            # 循环模式
            if not args.silent:
                logger.info("🔄 进入交互循环模式")
                logger.info("💡 输入提示词开始生图，输入 'exit'、'quit' 或按 Ctrl+C 退出")
                print("=" * 60)
            
            image_counter = 1
            
            while True:
                try:
                    # 获取用户输入
                    if args.prompt and image_counter == 1:
                        # 第一次使用命令行提供的prompt
                        prompt = args.prompt
                        if not args.silent:
                            print(f"📝 使用命令行提示词: {prompt}")
                    else:
                        prompt = input(f"\n[{image_counter}] 请输入提示词: ").strip()
                    
                    # 检查退出条件
                    if prompt.lower() in ['exit', 'quit', '']:
                        if not args.silent:
                            logger.info("👋 退出交互模式")
                        break
                    
                    # 生成图片
                    if not args.silent:
                        logger.info(f"🎨 开始生成第{image_counter}张图片...")
                    
                    output_path, actual_seed, gen_time = generate_single_image(
                        pipeline, prompt, args.width, args.height, args.steps, 
                        args.guidance, args.seed, args.quality, args.output, 
                        args.silent, image_counter
                    )
                    
                    if not args.silent:
                        logger.info(f"✅ 第{image_counter}张图片生成完成!")
                        logger.info(f"📸 图片已保存到: {output_path.absolute()}")
                        logger.info(f"🔢 使用种子: {actual_seed}")
                        logger.info(f"⏱️  生成耗时: {gen_time:.1f}s")
                    else:
                        print(f"Generated #{image_counter}: {output_path.absolute()} (seed: {actual_seed}, {gen_time:.1f}s)")
                    
                    image_counter += 1
                    
                except KeyboardInterrupt:
                    if not args.silent:
                        logger.info("\n👋 退出交互模式")
                    break
                except EOFError:
                    if not args.silent:
                        logger.info("\n👋 退出交互模式")
                    break
        else:
            # 单次生成模式
            if not args.silent:
                logger.info("🎨 开始生成图片...")
            
            output_path, actual_seed, gen_time = generate_single_image(
                pipeline, args.prompt, args.width, args.height, args.steps,
                args.guidance, args.seed, args.quality, args.output, args.silent
            )
            
            total_time = time.time() - load_start
            
            if not args.silent:
                logger.info(f"✅ 生成完成!")
                logger.info(f"📸 图片已保存到: {output_path.absolute()}")
                logger.info(f"🔢 使用种子: {actual_seed}")
                logger.info(f"⏱️  总耗时: {total_time:.1f}s (加载: {load_time:.1f}s + 生成: {gen_time:.1f}s)")
            else:
                print(f"Generated: {output_path.absolute()} (seed: {actual_seed}, {gen_time:.1f}s)")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 生成失败: {str(e)}")
        if not args.silent:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main()) 