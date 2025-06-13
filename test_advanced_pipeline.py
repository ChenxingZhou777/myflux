#!/usr/bin/env python3
"""
FLUX高级流水线分组测试脚本
测试transformer层级分组和流水线优化的效果
"""

import os
import sys
import time
import torch
from pathlib import Path

# 确保能够导入flux模块
sys.path.insert(0, str(Path(__file__).parent / "src"))

from flux.cli import print_gpu_memory
from flux.util import configs, load_flow_model
from flux.pipeline_optimizer import FluxModelGrouper, FluxPipelineExecutor


def test_model_grouping():
    """测试模型分组功能"""
    print("=" * 80)
    print("🧪 测试FLUX模型分组功能")
    print("=" * 80)
    
    # 检查设备
    if not torch.cuda.is_available():
        print("❌ 需要CUDA设备来运行测试")
        return False
    
    device = torch.device("cuda")
    print(f"🎯 使用设备: {device}")
    print_gpu_memory("测试开始")
    
    try:
        # 加载模型（仅用于分析结构）
        print("📦 加载FLUX模型进行分析...")
        model = load_flow_model("flux-dev", device="cpu")  # 先加载到CPU避免显存占用
        
        print(f"✅ 模型加载成功")
        print(f"   Double blocks: {len(model.double_blocks)}")
        print(f"   Single blocks: {len(model.single_blocks)}")
        
        # 创建分组器
        print("\n🏗️ 创建模型分组器...")
        grouper = FluxModelGrouper(model, device)
        
        # 打印分组策略
        grouper.print_grouping_strategy()
        
        # 分析分组结果
        print("\n📊 分组分析:")
        total_groups = len(grouper.groups)
        foundation_groups = len([g for g in grouper.groups.keys() if g == "foundation"])
        double_groups = len([g for g in grouper.groups.keys() if g.startswith("double_blocks")])
        single_groups = len([g for g in grouper.groups.keys() if g.startswith("single_blocks")])
        
        print(f"   总组数: {total_groups}")
        print(f"   基础组: {foundation_groups}")
        print(f"   Double block组: {double_groups}")
        print(f"   Single block组: {single_groups}")
        print(f"   其他组: {total_groups - foundation_groups - double_groups - single_groups}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_executor():
    """测试流水线执行器"""
    print("\n" + "=" * 80)
    print("🚀 测试流水线执行器")
    print("=" * 80)
    
    device = torch.device("cuda")
    print_gpu_memory("执行器测试开始")
    
    try:
        # 加载模型
        print("📦 加载FLUX模型...")
        model = load_flow_model("flux-dev", device="cpu")
        
        # 创建流水线执行器
        print("🔧 创建流水线执行器...")
        executor = FluxPipelineExecutor(model, device, max_workers=3)
        
        # 测试分组状态管理
        print("🔍 测试分组状态管理...")
        print(f"   初始状态: {executor.group_states}")
        
        # 测试依赖检查
        print("📋 测试依赖检查...")
        can_load_foundation = executor._can_load_group("foundation")
        print(f"   可以加载foundation: {can_load_foundation}")
        
        # 找一个依赖foundation的组进行测试
        dependent_group = None
        for group_name, group in executor.grouper.groups.items():
            if "foundation" in group.dependencies:
                dependent_group = group_name
                break
        
        if dependent_group:
            can_load_dependent = executor._can_load_group(dependent_group)
            print(f"   可以加载{dependent_group}: {can_load_dependent}")
        
        # 测试调度逻辑
        print("⚙️ 测试调度逻辑...")
        executor._schedule_next_loads()
        print(f"   调度后状态: {list(executor.group_states.keys())[:5]}...")  # 只显示前5个
        
        # 清理
        executor.cleanup()
        print("✅ 流水线执行器测试完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 执行器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_comparison():
    """测试性能对比（仅模拟，不做实际推理）"""
    print("\n" + "=" * 80)
    print("📈 性能对比分析（模拟）")
    print("=" * 80)
    
    try:
        device = torch.device("cuda")
        model = load_flow_model("flux-dev", device="cpu")
        grouper = FluxModelGrouper(model, device)
        
        # 计算预估性能
        total_load_time = sum(group.estimated_load_time for group in grouper.groups.values())
        total_compute_time = sum(group.estimated_compute_time for group in grouper.groups.values())
        
        # 模拟不同模式的性能
        sequential_time = total_load_time + total_compute_time
        
        # 流水线理论最佳时间（假设完美重叠）
        pipeline_time = max(total_load_time / 3, total_compute_time, 2.0)  # 至少2秒
        
        # 实际流水线时间（考虑开销）
        realistic_pipeline_time = pipeline_time * 1.2  # 20%开销
        
        print(f"📊 性能预估:")
        print(f"   顺序执行模式: {sequential_time:.1f}s")
        print(f"   理论流水线最佳: {pipeline_time:.1f}s")
        print(f"   实际流水线预期: {realistic_pipeline_time:.1f}s")
        print(f"   理论加速比: {sequential_time / pipeline_time:.1f}x")
        print(f"   实际加速比: {sequential_time / realistic_pipeline_time:.1f}x")
        
        # 分析瓶颈
        if total_load_time > total_compute_time:
            print(f"🔥 性能瓶颈: 模型加载 ({total_load_time:.1f}s vs {total_compute_time:.1f}s计算)")
            print("   建议: 增加并行加载线程数")
        else:
            print(f"🔥 性能瓶颈: 推理计算 ({total_compute_time:.1f}s vs {total_load_time:.1f}s加载)")
            print("   建议: 优化计算效率或使用更快的GPU")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能分析失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 FLUX高级流水线分组测试")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name()}")
        print_gpu_memory("测试开始前")
    
    # 运行所有测试
    tests = [
        ("模型分组功能", test_model_grouping),
        ("流水线执行器", test_pipeline_executor), 
        ("性能对比分析", test_performance_comparison),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"✅ {test_name} 测试{'成功' if success else '失败'}")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 80)
    print("📋 测试总结")
    print("=" * 80)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！高级流水线分组功能准备就绪。")
        print("\n使用方法:")
        print("python -m flux.cli --advanced_pipeline=true --pipeline_workers=3")
    else:
        print("⚠️ 部分测试失败，请检查问题后重试。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 