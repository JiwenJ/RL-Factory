
from PIL import Image, ImageDraw
from mcp.server.fastmcp import FastMCP
from io import BytesIO
import base64

# Initialize MCP server
mcp = FastMCP("ImageFlipServer")


@mcp.tool()
def rotate(img_base64: str=None, degree: int=None) -> str:
    """Rotate a Pillow image by specified degrees
    
    Args:
        img_base64: Input image as base64 encoded string
        degree: Rotation angle in degrees (positive for clockwise, negative for counterclockwise)
        
    Returns:
        str: Rotated image as base64 encoded string
    """
    print("================= call image_rotate tool ==================")
    # Convert base64 string to PIL Image
    img_data = base64.b64decode(img_base64)
    img = Image.open(BytesIO(img_data))
    
    # Determine rotation angle based on instruction
    angle = 0
    if degree:
        angle = degree
    
    # Rotate the image
    rotated_img = img.rotate(angle, expand=True)
    
    # Convert PIL Image back to base64 string
    buffer = BytesIO()
    rotated_img.save(buffer, format='PNG')
    img_base64_output = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64_output

if __name__ == "__main__":
    print("\nStarting MCP Image Editing Service...")
    mcp.run(transport='stdio')


    



