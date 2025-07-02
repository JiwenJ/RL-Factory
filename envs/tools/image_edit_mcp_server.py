# from PIL import Image, ImageDraw
# from mcp.server.fastmcp import FastMCP
# from io import BytesIO
# import base64

# Initialize MCP server
# mcp = FastMCP("ImageFlipServer")

# @mcp.tool()
# def image_flip(img_base64: str=None, instruction:str=None) -> str:
#     """Flip a Pillow image upside down based on natural language instructions
    
#     Args:
#         img_base64: Input image as base64 encoded string
#         instruction: Natural language editing instructions (ignored here)
        
#     Returns:
#         str: Flipped image as base64 encoded string
#     """
#     print("================= call image_flip tool ==================")
#     # Convert base64 string to PIL Image
#     img_data = base64.b64decode(img_base64)
#     img = Image.open(BytesIO(img_data))
    
#     # Flip the image upside down
#     flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
#     # Convert PIL Image back to base64 string
#     buffer = BytesIO()
#     flipped_img.save(buffer, format='PNG')
#     img_base64_output = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
#     return img_base64_output


# if __name__ == "__main__":
#     print("\nStarting MCP Image Flipping Service...")
#     mcp.run(transport='stdio')

from PIL import Image, ImageDraw
from mcp.server.fastmcp import FastMCP
from io import BytesIO
import base64

# # Initialize MCP server
mcp = FastMCP("ImageEditServer")
@mcp.tool()
def image_flip(img_base64: str=None, instruction: str=None) -> str:
    """Flip or rotate a Pillow image based on natural language instructions
    
    Args:
        img_base64: Input image as base64 encoded string
        instruction: Natural language editing instructions (top, down, left, right)
        
    Returns:
        str: Rotated image as base64 encoded string
    """
    print("================= call image_flip tool ==================")
    # Convert base64 string to PIL Image
    img_data = base64.b64decode(img_base64)
    img = Image.open(BytesIO(img_data))
    
    # Determine rotation angle based on instruction
    angle = 0
    if instruction:
        instruction = instruction.lower().strip()
        if instruction == 'top':
            angle = 0
        elif instruction == 'down':
            angle = 180
        elif instruction == 'left':
            angle = 270
        elif instruction == 'right':
            angle = 90
    
    # Rotate the image
    rotated_img = img.rotate(angle, expand=True)
    
    # Convert PIL Image back to base64 string
    buffer = BytesIO()
    rotated_img.save(buffer, format='PNG')
    img_base64_output = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64_output
# def image_edit(img_base64: str=None, instruction: str=None) -> str:
#     """Edit a Pillow image based on natural language instructions
    
#     Args:
#         img_base64: Input image as base64 encoded string
#         instruction: Natural language editing instructions
        
#     Returns:
#         str: Edited image as base64 encoded string
#     """
#     print("================= call image_edit tool ==================")
#     # Convert base64 string to PIL Image
#     img_data = base64.b64decode(img_base64)
#     img = Image.open(BytesIO(img_data))
    
#     # TODO: Add actual image editing logic based on instruction here
#     # For now, just return the original image
    
#     # Convert PIL Image back to base64 string
#     buffer = BytesIO()
#     img.save(buffer, format='PNG')
#     img_base64_output = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
#     return img_base64_output

if __name__ == "__main__":
    print("\nStarting MCP Image Editing Service...")
    mcp.run(transport='stdio')


    



