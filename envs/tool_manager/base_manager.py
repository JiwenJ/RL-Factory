from typing import List
from abc import ABC, abstractmethod


class ToolManager(ABC):
    def __init__(self, verl_config) -> None:
        self.verl_config = verl_config
        self.tool_map = {}
        self._build_tools()
    
    def get_tool(self, name_or_short_name: str):
        """通过名称或简写获取工具
        
        Args:
            name_or_short_name: 工具名称或简写
            
        Returns:
            找到的工具，如果没找到则返回None
        """
        name_or_short_name = str(name_or_short_name)
        return self.tool_map.get(name_or_short_name, None)
    
    @property
    @abstractmethod
    def all_tools(self):
        raise NotImplementedError

    @abstractmethod
    def _build_tools(self):
        raise NotImplementedError
    
    @abstractmethod
    def execute_actions(self, responses: List[str]):
        raise NotImplementedError
