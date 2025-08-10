import hashlib
import json
import time
from typing import Any, Optional, Dict
from cachebox import Cache, LRUCache, LFUCache, FIFOCache
from .cache_base import CacheBase, CacheMode, EvictionPolicy
from functools import wraps

class CacheBoxCache(CacheBase):
    def __init__(self, max_size: int = 1000, mode: CacheMode = CacheMode.SINGLE,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        """初始化缓存
        
        Args:
            max_size: 最大缓存条目数
            mode: 缓存模式
            eviction_policy: 缓存淘汰策略
        """
        self.mode = mode
        self.set_eviction_policy(eviction_policy, max_size)
        self.stats = {
            "hits": 0,
            "misses": 0,
            "size": 0,
            "evictions": 0,
            "ttl_expirations": 0
        }
        self.ttl_map = {}  # 用于存储TTL信息
    
    def _hash_key(self, method_name: str, params: dict) -> str:
        """生成缓存键
        
        Args:
            method_name: 方法名
            params: 参数字典
            
        Returns:
            str: 哈希后的键
        """
        key_str = f"{method_name}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        # 检查TTL
        if key in self.ttl_map:
            if time.time() > self.ttl_map[key]:
                del self.ttl_map[key]
                self.stats["ttl_expirations"] += 1
                return None
        
        value = self.cache.get(key)
        if value is not None:
            self.stats["hits"] += 1
        else:
            self.stats["misses"] += 1
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self.cache[key] = value
        self.stats["size"] = len(self.cache)
        
        # 设置TTL
        if ttl is not None:
            self.ttl_map[key] = time.time() + ttl
    
    def delete(self, key: str) -> None:
        if key in self.cache:
            del self.cache[key]
        if key in self.ttl_map:
            del self.ttl_map[key]
        self.stats["size"] = len(self.cache)
    
    def clear(self) -> None:
        self.cache.clear()
        self.ttl_map.clear()
        self.stats["size"] = 0
    
    def has(self, key: str) -> bool:
        if key in self.ttl_map and time.time() > self.ttl_map[key]:
            del self.ttl_map[key]
            return False
        return key in self.cache
    
    def get_mode(self) -> CacheMode:
        return self.mode
    
    def get_stats(self) -> Dict[str, Any]:
        return self.stats
    
    def get_eviction_policy(self) -> EvictionPolicy:
        return self.eviction_policy
    
    def set_eviction_policy(self, policy: EvictionPolicy, max_size: int) -> None:
        self.eviction_policy = policy
        # TODO: 实现不同淘汰策略的具体逻辑
        if policy == EvictionPolicy.LRU:
            self.cache = LRUCache(maxsize=max_size)
        elif policy == EvictionPolicy.LFU:
            self.cache = LFUCache(maxsize=max_size)
        elif policy == EvictionPolicy.FIFO:
            self.cache = FIFOCache(maxsize=max_size)
        else:
            self.cache = Cache(maxsize=max_size)

    def cache_decorator(self):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 从args和kwargs生成一个稳定的key
                # 注意：这需要args和kwargs中的值是可哈希的
                key_parts = [func.__name__] + list(args) + sorted(kwargs.items())
                key = self._hash_key(func.__name__, {'args': args, 'kwargs': sorted(kwargs.items())})

                if self.has(key):
                    return self.get(key)
                
                result = await func(*args, **kwargs)
                self.set(key, result)
                return result
            return wrapper
        return decorator 