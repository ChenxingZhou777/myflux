#!/usr/bin/env python3
"""
Flux FP8 命令行直接生图工具

使用示例:
python generate_cli.py --config configs/config-dev-offload-1-4090.json \
                       --prompt "a beautiful landscape" \
                       --output my_image.jpg \
                       --width 1024 --height 1024 \
                       --steps 24 --guidance 3.5 --seed 12345
"""

import argparse
import time
from pathlib import Path
from flux_pipeline import FluxPipeline
from loguru import logger


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
        required=True,
        help="生图提示词"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="generated_image.jpg",
        help="输出图片路径 (默认: generated_image.jpg)"
    )
    
    # 可选参数
    parser.add_argument("--width", "-w", type=int, default=1024, help="图片宽度 (默认: 1024)")
    parser.add_argument("--height", "-H", type=int, default=1024, help="图片高度 (默认: 1024)")
    parser.add_argument("--steps", "-s", type=int, default=24, help="推理步数 (默认: 24)")
    parser.add_argument("--guidance", "-g", type=float, default=3.5, help="引导强度 (默认: 3.5)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (默认: 随机)")
    parser.add_argument("--quality", "-q", type=int, default=95, help="JPEG质量 (默认: 95)")
    parser.add_argument("--silent", action="store_true", help="静默模式，不显示详细信息")
    
    args = parser.parse_args()
    
    
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
        
        # 生成图片
        if not args.silent:
            logger.info("🎨 开始生成图片...")
        
        gen_start = time.time()
        result, seed = pipeline.generate(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            num_steps=args.steps,
            guidance=args.guidance,
            seed=args.seed,
            silent=args.silent,
            return_seed=True,
            jpeg_quality=args.quality
        )
        gen_time = time.time() - gen_start
        
        # 保存图片
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(result.getvalue())
        
        total_time = time.time() - load_start
        
        if not args.silent:
            logger.info(f"✅ 生成完成!")
            logger.info(f"📸 图片已保存到: {output_path.absolute()}")
            logger.info(f"🔢 使用种子: {seed}")
            logger.info(f"⏱️  总耗时: {total_time:.1f}s (加载: {load_time:.1f}s + 生成: {gen_time:.1f}s)")
        else:
            print(f"Generated: {output_path.absolute()} (seed: {seed}, {gen_time:.1f}s)")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 生成失败: {str(e)}")
        if not args.silent:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main()) 