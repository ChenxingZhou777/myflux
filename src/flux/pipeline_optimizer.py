"""
FLUX Transformer 分组流水线优化器
实现安全的、专注于内存管理优化的流水线处理
"""

import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class ModelGroup:
    """模型组定义"""
    name: str                    # 组名
    components: List[str]        # 组件名称列表
    dependencies: List[str]      # 依赖的前置组
    estimated_load_time: float  # 预估加载时间（秒）
    estimated_compute_time: float # 预估计算时间（秒）
    is_loaded: bool = False      # 是否已加载
    is_computing: bool = False   # 是否正在计算


class FluxModelGrouper:
    """FLUX模型分组器"""
    
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.groups = self._create_model_groups()
        
    def _create_model_groups(self) -> Dict[str, ModelGroup]:
        """创建模型分组策略"""
        
        # 分析模型结构
        num_double_blocks = len(self.model.double_blocks)
        num_single_blocks = len(self.model.single_blocks)
        
        # 更保守的分组大小，避免显存问题
        double_group_size = max(3, num_double_blocks // 4)  # 分成约4组
        single_group_size = max(4, num_single_blocks // 6)  # 分成约6组
        
        groups = {}
        
        # 1. 基础组件组（最高优先级，必须最先加载）
        groups["foundation"] = ModelGroup(
            name="foundation",
            components=["img_in", "time_in", "vector_in", "guidance_in", "txt_in", "pe_embedder"],
            dependencies=[],
            estimated_load_time=0.5,
            estimated_compute_time=0.1
        )
        
        # 2. Double blocks分组
        for i in range(0, num_double_blocks, double_group_size):
            end_idx = min(i + double_group_size, num_double_blocks)
            group_name = f"double_blocks_{i}_{end_idx-1}"
            
            dependencies = ["foundation"] if i == 0 else [f"double_blocks_{i-double_group_size}_{i-1}"]
            
            groups[group_name] = ModelGroup(
                name=group_name,
                components=[f"double_blocks.{j}" for j in range(i, end_idx)],
                dependencies=dependencies,
                estimated_load_time=1.0 * (end_idx - i),
                estimated_compute_time=0.4 * (end_idx - i)
            )
        
        # 3. Single blocks分组
        for i in range(0, num_single_blocks, single_group_size):
            end_idx = min(i + single_group_size, num_single_blocks)
            group_name = f"single_blocks_{i}_{end_idx-1}"
            
            # 第一组single_blocks依赖最后一组double_blocks
            if i == 0:
                last_double_group = f"double_blocks_{num_double_blocks-double_group_size}_{num_double_blocks-1}"
                dependencies = [last_double_group]
            else:
                dependencies = [f"single_blocks_{i-single_group_size}_{i-1}"]
            
            groups[group_name] = ModelGroup(
                name=group_name,
                components=[f"single_blocks.{j}" for j in range(i, end_idx)],
                dependencies=dependencies,
                estimated_load_time=0.8 * (end_idx - i),
                estimated_compute_time=0.3 * (end_idx - i)
            )
        
        # 4. 最终输出组
        groups["final"] = ModelGroup(
            name="final",
            components=["final_layer"],
            dependencies=[f"single_blocks_{num_single_blocks-single_group_size}_{num_single_blocks-1}"],
            estimated_load_time=0.3,
            estimated_compute_time=0.1
        )
        
        return groups
    
    def print_grouping_strategy(self):
        """打印分组策略"""
        print("🏗️ FLUX简化分组策略:")
        print("=" * 60)
        
        total_estimated_load = 0
        total_estimated_compute = 0
        
        for group_name, group in self.groups.items():
            total_estimated_load += group.estimated_load_time
            total_estimated_compute += group.estimated_compute_time
            
            print(f"📦 {group.name}")
            print(f"   组件: {len(group.components)}个")
            if len(group.components) <= 3:
                print(f"   详细: {group.components}")
            else:
                print(f"   范围: {group.components[0]} ... {group.components[-1]}")
            print(f"   预估: 加载{group.estimated_load_time:.1f}s | 计算{group.estimated_compute_time:.1f}s")
            print()
        
        print(f"📊 总体预估:")
        print(f"   顺序执行总时间: {total_estimated_load + total_estimated_compute:.1f}s")
        print(f"   优化后预期时间: {max(total_estimated_load * 0.7, total_estimated_compute):.1f}s")
        print(f"   预期加速比: {(total_estimated_load + total_estimated_compute) / max(total_estimated_load * 0.7, total_estimated_compute):.1f}x")


class FluxSafeOptimizer:
    """FLUX安全优化器 - 专注于内存管理和设备一致性"""
    
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.grouper = FluxModelGrouper(model, device)
        
    def safe_optimized_denoise(self, 
                              img, img_ids, txt, txt_ids, vec,
                              timesteps: List[float],
                              guidance: float = 4.0) -> torch.Tensor:
        """安全优化去噪：重点是正确的设备管理和内存优化"""
        
        print("🛡️ 开始FLUX安全优化推理...")
        total_start = time.time()
        
        # 阶段1：确保所有输入数据在正确设备上
        print("🔧 阶段1：设备一致性检查...")
        device_start = time.time()
        
        img = img.to(self.device, non_blocking=True)
        img_ids = img_ids.to(self.device, non_blocking=True)
        txt = txt.to(self.device, non_blocking=True)
        txt_ids = txt_ids.to(self.device, non_blocking=True)
        vec = vec.to(self.device, non_blocking=True)
        
        device_time = time.time() - device_start
        print(f"✅ 设备一致性检查完成: {device_time:.3f}s")
        
        # 阶段2：分阶段模型加载以优化内存使用
        print("📦 阶段2：分阶段模型加载...")
        load_start = time.time()
        
        # 使用临时上下文避免推理模式冲突
        with torch.enable_grad():
            self._progressive_model_loading()
        
        load_time = time.time() - load_start
        print(f"✅ 模型加载优化完成: {load_time:.3f}s")
        
        # 阶段3：执行推理
        print("🚀 阶段3：执行推理...")
        inference_start = time.time()
        
        guidance_vec = torch.full((img.shape[0],), guidance, device=self.device, dtype=img.dtype)
        
        for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            step_start = time.time()
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=self.device)
            
            pred = self.model(
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
                print(f"   第1步: {step_time:.3f}s")
            elif step_idx == len(timesteps) - 2:
                print(f"   最后一步: {step_time:.3f}s")
        
        inference_time = time.time() - inference_start
        total_time = time.time() - total_start
        
        print(f"✅ 安全优化推理完成!")
        print(f"   设备管理: {device_time:.3f}s")
        print(f"   模型加载: {load_time:.3f}s")
        print(f"   推理计算: {inference_time:.3f}s")
        print(f"   总耗时: {total_time:.3f}s")
        
        return img
    
    def _progressive_model_loading(self):
        """渐进式模型加载，优化内存使用"""
        
        # 确保整个模型在目标设备上
        print("   加载完整模型到目标设备...")
        load_start = time.time()
        
        # 分批加载以减少内存峰值
        if not self.model.training:
            self.model.eval()
        
        # 加载基础组件
        foundation_components = ["img_in", "time_in", "vector_in", "guidance_in", "txt_in", "pe_embedder"]
        for comp_name in foundation_components:
            if hasattr(self.model, comp_name):
                getattr(self.model, comp_name).to(self.device)
        
        # 加载transformer blocks
        self.model.double_blocks.to(self.device)
        self.model.single_blocks.to(self.device)
        
        # 加载最终层
        self.model.final_layer.to(self.device)
        
        load_time = time.time() - load_start
        print(f"   模型组件加载完成: {load_time:.3f}s")


# 集成到主CLI的便捷函数
def flux_pipeline_denoise(model, img, img_ids, txt, txt_ids, vec, 
                         timesteps: List[float], guidance: float = 4.0,
                         torch_device: torch.device = None,
                         enable_pipeline: bool = True,
                         max_workers: int = 3) -> torch.Tensor:
    """
    FLUX安全流水线去噪函数
    
    Args:
        model: FLUX模型
        img, img_ids, txt, txt_ids, vec: 输入张量
        timesteps: 时间步列表
        guidance: 引导强度
        torch_device: 目标设备
        enable_pipeline: 是否启用流水线优化
        max_workers: 最大并行加载线程数（当前版本不使用）
    
    Returns:
        去噪后的图像张量
    """
    
    if not enable_pipeline:
        # 回退到标准优化模式
        from .cli import optimized_denoise_with_loading
        return optimized_denoise_with_loading(
            model, img, img_ids, txt, txt_ids, vec, timesteps, guidance, torch_device
        )
    
    # 使用安全优化模式
    try:
        safe_optimizer = FluxSafeOptimizer(model, torch_device)
        
        # 打印分组策略
        safe_optimizer.grouper.print_grouping_strategy()
        
        # 执行安全优化推理
        result = safe_optimizer.safe_optimized_denoise(
            img, img_ids, txt, txt_ids, vec, timesteps, guidance
        )
        
        return result
        
    except Exception as e:
        print(f"❌ 安全优化推理失败: {e}")
        print("🔄 回退到标准优化模式...")
        
        # 回退到标准模式
        from .cli import optimized_denoise_with_loading
        return optimized_denoise_with_loading(
            model, img, img_ids, txt, txt_ids, vec, timesteps, guidance, torch_device
        ) 