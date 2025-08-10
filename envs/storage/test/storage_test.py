import asyncio
import argparse
import time
import os
import shutil
import hashlib
from datetime import datetime
from typing import Optional
from envs.storage.manager.storage_manager import create_storage_manager, parse_args

async def test_storage_manager():
    # 解析命令行参数
    args = parse_args()
    
    print(f"--- 缓存管理器测试 ---")
    print(f"缓存模式: {args.cache}")
    print(f"持久化: {args.persist}")
    print(f"淘汰策略: {args.eviction}")
    print(f"最大缓存大小: {args.max_size}")
    print(f"持久化目录: {args.persist_dir}")
    print(f"加载缓存文件: {args.load_cache}")
    print("-" * 50)
    
    # 确保持久化目录存在
    if args.persist:
        os.makedirs(args.persist_dir, exist_ok=True)
        print(f"持久化目录已创建: {args.persist_dir}")
    
    # 创建存储管理器
    storage = create_storage_manager(args)
    
    # 如果指定了load_cache，给系统一点时间来加载，并检查是否成功
    if args.load_cache:
        print(f"\n--- 测试缓存加载功能 ---")
        load_cache_path = args.load_cache
        if not os.path.isabs(load_cache_path):
            load_cache_path = os.path.join(args.persist_dir, args.load_cache)
        
        if os.path.exists(load_cache_path):
            print(f"缓存路径: {load_cache_path}")
            print("等待缓存加载...")
            await asyncio.sleep(0.5)
            
            # 测试加载的缓存是否可用 - 尝试多个可能的缓存键
            print("\n测试加载的缓存数据可用性...")
            test_methods = [
                ("search_api", {"query": "test query", "page": 1}),
                ("user_profile", {"user_id": 12345}),
                ("persist_test_1", {"data": "value1"}),
                ("persist_test_2", {"data": "value2"}),
            ]
            
            found_cached_data = False
            for method_name, params in test_methods:
                result = await storage.get(method_name, params)
                if result:
                    print(f"✓ 从缓存中成功获取 {method_name} 的数据:")
                    print(f"   参数: {params}")
                    print(f"   结果: {result}")
                    found_cached_data = True
                    break
            
            if not found_cached_data:
                print("✗ 未能从加载的缓存中获取任何数据")
        else:
            print(f"⚠ 指定的缓存路径不存在: {load_cache_path}")
    
    # 模拟一个工具调用
    async def mock_tool_call(method_name: str, params: dict, ttl: Optional[int] = None):
        # 尝试从存储获取
        result = await storage.get(method_name, params)
        
        if result is not None:
            print(f"Cache hit for {method_name}")
            return result
        
        # 模拟实际调用
        print(f"Cache miss for {method_name}, executing...")
        await asyncio.sleep(0.1)  # 模拟0.1秒的耗时操作
        result = {"result": f"Result for {method_name} with params {params}", "timestamp": datetime.now().isoformat()}
        
        # 存入存储
        await storage.set(method_name, params, result, ttl)
        
        return result

    print(f"\n--- 基础缓存功能测试 ---")
    
    # 测试基础缓存功能
    test_cases = [
        ("search_api", {"query": "test query", "page": 1}, None),
        ("search_api", {"query": "test query", "page": 1}, None),  # 应该命中缓存
        ("user_profile", {"user_id": 12345}, 60),  # 带TTL
        ("user_profile", {"user_id": 12345}, 60),  # 应该命中缓存
    ]
    
    for i, (method_name, params, ttl) in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {method_name} with {params}")
        start_time = time.perf_counter()
        result = await mock_tool_call(method_name, params, ttl)
        duration = time.perf_counter() - start_time
        print(f"耗时: {duration:.4f}秒")

    # 如果启用了持久化，测试持久化功能
    if args.persist:
        print(f"\n--- 持久化功能测试 ---")
        
        # 添加一些测试数据
        test_data = [
            ("persist_test_1", {"data": "value1"}, {"result": "Persisted data 1", "created": datetime.now().isoformat()}),
            ("persist_test_2", {"data": "value2"}, {"result": "Persisted data 2", "created": datetime.now().isoformat()}),
        ]
        
        for method_name, params, result in test_data:
            await storage.set(method_name, params, result)
            print(f"已添加测试数据: {method_name}")
        
        # 手动触发持久化
        print("\n触发持久化保存...")
        await storage._flush_persist_buffer()
        
        # 等待文件写入完成
        await asyncio.sleep(0.2)
        
        # 检查持久化文件是否创建
        if storage.persist:
            cache_files = storage.persist.list_cache_files()
            print(f"持久化文件列表:")
            for file_path in cache_files:
                file_size = os.path.getsize(file_path)
                print(f"  - {file_path} ({file_size} bytes)")
        
        # 测试从持久化存储重新加载
        print("\n测试从持久化存储重新加载...")
        
        # 清空内存缓存
        storage.cache.clear()
        print("内存缓存已清空")
        
        # 从持久化存储加载数据
        for method_name, params, expected_result in test_data:
            result = await storage.get(method_name, params)
            if result:
                print(f"✓ 成功从持久化存储加载: {method_name}")
            else:
                print(f"✗ 未能从持久化存储加载: {method_name}")

    # 打印最终统计信息
    print(f"\n--- 最终统计信息 ---")
    stats = storage.get_stats()
    print(f"缓存模式: {stats['cache_mode']}")
    print(f"持久化启用: {stats['enable_persist']}")
    print(f"淘汰策略: {stats['eviction_policy']}")
    print(f"缓存统计: {stats['cache_stats']}")
    if stats['enable_persist']:
        print(f"持久化缓冲区大小: {stats['persist_buffer_size']}")
        print(f"上次持久化时间: {stats['last_persist_time']}")
    
    # 模拟训练结束，优雅关闭存储管理器
    print(f"\n--- 训练结束，关闭存储管理器 ---")
    await storage.shutdown()
    
    print("\n测试完成！")

if __name__ == "__main__":
    asyncio.run(test_storage_manager())