import asyncio
from typing import Dict
from contextlib import asynccontextmanager

class ConcurrencyLimiter:
    """简单的并发限制器"""
    
    def __init__(self, global_limit: int = 100):
        # 工具级别的信号量
        self._tool_semaphores: Dict[str, asyncio.Semaphore] = {}
        # 全局信号量
        self._global_semaphore = asyncio.Semaphore(global_limit)
        
    def get_tool_semaphore(self, tool_name: str, limit: int = 10) -> asyncio.Semaphore:
        """获取或创建工具的信号量"""
        if tool_name not in self._tool_semaphores:
            self._tool_semaphores[tool_name] = asyncio.Semaphore(limit)
        return self._tool_semaphores[tool_name]
        
    @asynccontextmanager
    async def limit(self, tool_name: str):
        """使用上下文管理器控制并发"""
        sem = self.get_tool_semaphore(tool_name)
        async with self._global_semaphore:
            async with sem:
                yield
