import asyncio
import sys
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path):
        command = "python" if server_script_path.endswith('.py') else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        response = await self.session.list_tools()
        tools = response.tools
        print("Connected to server with tools:", [tool.name for tool in tools])

    async def call_tool(self, tool_name, input_data):
        result = await self.session.call_tool(tool_name, input_data)
        return result

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    client = MCPClient()
    try:
        await client.connect_to_server("/Users/jiwen/Github/RL-Factory/envs/tools/echo_server.py")
        # 假设有 greet 工具
        result = await client.call_tool("greet", {"name": "World"})
        breakpoint()
        print("Greet result:", result)
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
