import asyncio
import sys
import base64
import os
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class ImageEditClient:
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

    async def edit_image(self, image_path: str, instruction: str, output_path: str):
        """Edit an image using the server's PIL-based editing tool"""
        # Read image and convert to base64
        with open(image_path, "rb") as f:
            img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        # Call the image editing tool
        call_result = await self.session.call_tool("edit_image_pil", {
            "img_base64": img_base64,
            "instruction": instruction
        })
        # breakpoint()
        # Save edited image (access the result property)
        edited_img_data = base64.b64decode(call_result.content[0].text)
        with open(output_path, "wb") as f:
            f.write(edited_img_data)
        print(f"Edited image saved to {output_path}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    from PIL import Image
    input_path = "/Users/jiwen/Github/RL-Factory/envs/tools/bigmap.jpg"
    width, height = Image.open(input_path).size
    instruction = "crop region 0,{top},{right},{bottom} and resize to 2000x1500"
    output_path = "/Users/jiwen/Github/RL-Factory/envs/tools/bigmap_processed.png"
    print(f"Original size: {width}x{height}") 
    instruction = instruction.format(
        top=height // 2,
        right=width // 2,
        bottom=height
    )
    print(instruction)
    client = ImageEditClient()
    try:
        # Path to the image edit server script
        server_script_path = os.path.join(
            os.path.dirname(__file__), 
            "image_edit_mcp_server.py"
        )
        await client.connect_to_server(server_script_path)
        await client.edit_image(input_path, instruction, output_path)
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
