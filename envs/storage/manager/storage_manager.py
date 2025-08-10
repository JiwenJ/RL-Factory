import argparse
import asyncio
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from ..cache.cache_base import CacheMode, EvictionPolicy
from ..cache.cachebox_cache import CacheBoxCache
from ..persist.disk_persist import DiskPersist


class StorageManager:
    def __init__(self, cache_mode: CacheMode = CacheMode.SINGLE,
                 enable_persist: bool = False,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
                 max_cache_size: int = 1000,
                 persist_dir: str = "cache_data",
                 persist_interval: int = 300,  # 持久化间隔，默认5分钟
                 sync_interval: int = 600, # 同步间隔，默认10分钟
                 load_cache_path: Optional[str] = None):
        """初始化存储管理器
        
        Args:
            cache_mode: 缓存模式
            enable_persist: 是否启用持久化
            eviction_policy: 缓存淘汰策略
            max_cache_size: 最大缓存条目数
            persist_dir: 持久化存储目录
            persist_interval: 持久化间隔（秒）
            sync_interval: 同步间隔（秒）
        """
        self.cache_mode = cache_mode
        self.enable_persist = enable_persist
        self.persist_interval = persist_interval
        self.sync_interval = sync_interval

        # 初始化缓存
        self.cache = CacheBoxCache(
            max_size=max_cache_size,
            mode=cache_mode,
            eviction_policy=eviction_policy
        )

        # 如果启用持久化，初始化持久化存储
        self.persist = DiskPersist(base_dir=persist_dir) if enable_persist else None

        # 用于批量写入的缓冲区
        self.persist_buffer: Dict[str, Dict[str, Any]] = {}
        self.last_persist_time = datetime.now()

        # 如果启用持久化，启动定时任务
        if self.enable_persist and self.persist:
            if load_cache_path:
                asyncio.create_task(self.load_initial_cache(load_cache_path))
            asyncio.create_task(self._start_persist_task())
            asyncio.create_task(self._start_sync_task())

    async def load_initial_cache(self, path: str):
        """从指定路径加载初始缓存"""
        if not self.persist:
            print(f"Warning: --load-cache specified, but persistence is disabled. Skipping.")
            return

        print(f"Attempting to load initial cache from: {path}")
        try:
            if os.path.isfile(path):
                files_to_load = [path]
            elif os.path.isdir(path):
                files_to_load = [os.path.join(path, f) for f in os.listdir(path) 
                               if os.path.isfile(os.path.join(path, f)) and f.endswith('.cache')]
            else:
                print(f"Error: Path specified by --load-cache is not a valid file or directory: {path}")
                return

            if not files_to_load:
                print("No cache files found to load.")
                return
            
            print(f"Found {len(files_to_load)} cache file(s) to load:")
            for file_path in files_to_load:
                print(f"  - {os.path.basename(file_path)}")
            
            loaded_count = 0
            total_entries = 0
            for file_path in files_to_load:
                try:
                    data = await asyncio.to_thread(self.persist.load, file_path)
                    entry_count = len(data)
                    
                    # 打印读取的数据内容
                    print(f"\n--- 从文件 {os.path.basename(file_path)} 读取的数据 ---")
                    for cache_key, value in data.items():
                        print(f"  缓存键: {cache_key}")
                        print(f"  缓存值: {value}")
                        print("  " + "-" * 40)
                        self.cache.set(cache_key, value)
                    
                    print(f"✓ Loaded {entry_count} entries from {os.path.basename(file_path)}")
                    loaded_count += 1
                    total_entries += entry_count
                except Exception as e:
                    print(f"✗ Failed to load cache from {os.path.basename(file_path)}: {e}")
            
            print(f"Cache loading summary: {loaded_count}/{len(files_to_load)} files loaded, {total_entries} total entries")

        except Exception as e:
            print(f"An error occurred while processing --load-cache path {path}: {e}")

    async def _start_persist_task(self):
        """启动定时持久化任务"""
        while True:
            await asyncio.sleep(self.persist_interval)
            await self._flush_persist_buffer()

    async def _start_sync_task(self):
        """启动定时同步任务"""
        while True:
            await asyncio.sleep(self.sync_interval)
            await self._sync_from_persist()

    async def _flush_persist_buffer(self):
        """将缓冲区数据写入持久化存储"""
        if not self.persist_buffer:
            return

        try:
            # 选项1：按方法名分别保存（当前行为）
            for method_name, data in self.persist_buffer.items():
                await asyncio.to_thread(self.persist.save, data, method_name)
            
            # 选项2：合并所有数据到一个文件（如果需要的话）
            # all_data = {}
            # for method_name, data in self.persist_buffer.items():
            #     all_data.update(data)
            # await asyncio.to_thread(self.persist.save, all_data, "all_cache")
            
            self.persist_buffer.clear()
            self.last_persist_time = datetime.now()
        except Exception as e:
            print(f"Failed to flush persist buffer: {e}")

    async def _sync_from_persist(self):
        """从持久化存储同步数据到缓存"""
        if not self.persist:
            return

        try:
            # 获取所有缓存文件
            cache_files = self.persist.list_cache_files()
            for file_path in cache_files:
                data = await asyncio.to_thread(self.persist.load, file_path)
                for cache_key, value in data.items():
                    # 使用缓存的hash_key方法确保一致性
                    self.cache.set(cache_key, value)
        except Exception as e:
            print(f"Failed to sync from persist: {e}")

    async def get(self, method_name: str, params: dict) -> Optional[Any]:
        """获取缓存数据
        
        Args:
            method_name: 方法名
            params: 参数字典
            
        Returns:
            Optional[Any]: 缓存数据
        """
        cache_key = self.cache._hash_key(method_name, params)

        # 尝试从缓存获取
        if self.cache.has(cache_key):
            return self.cache.get(cache_key)

        # 如果启用持久化，尝试从持久化存储加载
        if self.enable_persist and self.persist:
            try:
                # 查找相关的缓存文件
                cache_files = self.persist.list_cache_files(method_name)
                for file_path in cache_files:
                    data = await asyncio.to_thread(self.persist.load, file_path)
                    if cache_key in data:
                        value = data[cache_key]
                        # 写入缓存
                        self.cache.set(cache_key, value)
                        return value
            except Exception as e:
                print(f"Failed to load from persist: {e}")

        return None

    async def set(self, method_name: str, params: dict, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存数据
        
        Args:
            method_name: 方法名
            params: 参数字典
            value: 要缓存的值
            ttl: 过期时间（秒）
        """
        cache_key = self.cache._hash_key(method_name, params)

        # 写入缓存
        self.cache.set(cache_key, value, ttl)

        # 如果启用持久化，写入缓冲区
        if self.enable_persist and self.persist:
            if method_name not in self.persist_buffer:
                self.persist_buffer[method_name] = {}
            self.persist_buffer[method_name][cache_key] = value

            # 如果缓冲区过大或距离上次持久化时间过长，触发持久化
            current_time = datetime.now()
            if (len(self.persist_buffer) > 1000 or
                    (current_time - self.last_persist_time).total_seconds() > self.persist_interval):
                await self._flush_persist_buffer()

    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = {
            "cache_mode": self.cache_mode.value,
            "enable_persist": self.enable_persist,
            "eviction_policy": self.cache.get_eviction_policy().value,
            "cache_stats": self.cache.get_stats(),
            "persist_buffer_size": len(self.persist_buffer) if self.enable_persist else 0,
            "last_persist_time": self.last_persist_time.isoformat() if self.enable_persist else None
        }
        return stats

    async def shutdown(self):
        """优雅关闭存储管理器，确保所有数据被持久化"""
        if self.enable_persist and self.persist:
            print("正在关闭存储管理器，保存缓存数据...")
            
            # 强制刷新持久化缓冲区
            await self._flush_persist_buffer()
            
            # 等待所有异步任务完成
            await asyncio.sleep(0.1)
            
            print(f"缓存数据已保存到: {self.persist.base_dir}")
            
            # 显示保存的文件信息
            try:
                cache_files = self.persist.list_cache_files()
                if cache_files:
                    print("已保存的缓存文件:")
                    total_size = 0
                    for file_path in cache_files:
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        print(f"  - {os.path.basename(file_path)}: {file_size} bytes")
                    print(f"总缓存大小: {total_size} bytes")
                else:
                    print("没有缓存文件需要保存")
            except Exception as e:
                print(f"检查缓存文件时出错: {e}")
        else:
            print("持久化未启用，无需保存缓存数据")

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时自动关闭"""
        # 由于这是同步方法，我们需要在异步环境中处理
        # 建议在实际使用时手动调用 shutdown()
        pass

    async def __aenter__(self):
        """支持异步上下文管理器"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步退出上下文时自动关闭"""
        await self.shutdown()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Storage Manager")
    parser.add_argument("--cache", choices=["single", "multi"], default="single",
                        help="Cache mode: single (default) or multi")
    parser.add_argument("--persist", action="store_true",
                        help="Enable persistence")
    parser.add_argument("--eviction", choices=["lru", "lfu", "fifo", "ttl"], default="lru",
                        help="Cache eviction policy")
    parser.add_argument("--max-size", type=int, default=1000,
                        help="Maximum cache size")
    parser.add_argument("--persist-dir", default="cache_data",
                        help="Persistence directory")
    parser.add_argument("--load-cache", type=str, default=None,
                        help="Path to a cache file or directory to preload")
    return parser.parse_args()


def create_storage_manager(args) -> StorageManager:
    """创建存储管理器
    
    Args:
        args: 命令行参数
        
    Returns:
        StorageManager: 存储管理器实例
    """
    cache_mode = CacheMode.SINGLE if args.cache == "single" else CacheMode.MULTI
    eviction_policy = EvictionPolicy(args.eviction)

    return StorageManager(
        cache_mode=cache_mode,
        enable_persist=args.persist,
        eviction_policy=eviction_policy,
        max_cache_size=args.max_size,
        persist_dir=args.persist_dir,
        load_cache_path=args.load_cache
    )


def create_config_storage_manager(verl_config) -> StorageManager:
    """创建存储管理器

    Args:
        verl_config: 配置文件

    Returns:
        StorageManager: 存储管理器实例
    """
    cache_mode = CacheMode.SINGLE if verl_config.cache == "single" else CacheMode.MULTI
    eviction_policy = EvictionPolicy(verl_config.eviction)
    load_cache_path = getattr(verl_config, 'load_cache', None)

    return StorageManager(
        cache_mode=cache_mode,
        enable_persist=verl_config.persist,
        eviction_policy=eviction_policy,
        max_cache_size=verl_config.max_size,
        persist_dir=verl_config.persist_dir,
        load_cache_path=load_cache_path
    )
