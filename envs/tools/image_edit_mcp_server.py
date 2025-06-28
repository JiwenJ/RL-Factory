from PIL import Image, ImageDraw
from mcp.server.fastmcp import FastMCP
from io import BytesIO
import base64

# Initialize MCP server
mcp = FastMCP("ImageEditServer")
@mcp.tool()
def image_edit(img_base64: str=None, instruction: str=None) -> str:
    """Edit a Pillow image based on natural language instructions
    
    Args:
        img_base64: Input image as base64 encoded string
        instruction: Natural language editing instructions
        
    Returns:
        str: Edited image as base64 encoded string
    """
    print("================= call image_edit tool ==================")
    # Convert base64 string to PIL Image
    img_data = base64.b64decode(img_base64)
    img = Image.open(BytesIO(img_data))
    
    # TODO: Add actual image editing logic based on instruction here
    # For now, just return the original image
    
    # Convert PIL Image back to base64 string
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_base64_output = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64_output

if __name__ == "__main__":
    print("\nStarting MCP Image Editing Service...")
    mcp.run(transport='stdio')


    



