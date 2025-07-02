from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from mcp.server.fastmcp import FastMCP
from io import BytesIO
import base64
import re

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
    print(f"Instruction: {instruction}")
    
    # Convert base64 string to PIL Image
    img_data = base64.b64decode(img_base64)
    img = Image.open(BytesIO(img_data))
    original_size = img.size
    
    instruction_lower = instruction.lower() if instruction else ""
    
    # **Rotate functionality**
    if 'rotate' in instruction_lower:
        # Extract degrees from instruction, default to 90 if not found
        match = re.search(r'rotate\s*(\d+)', instruction_lower)
        degrees = int(match.group(1)) if match else 90
        img = img.rotate(degrees, expand=True)
        print(f"Rotated image by {degrees} degrees")
    
    # **Zoom in functionality**
    if 'zoom in' in instruction_lower or ('zoom' in instruction_lower and 'out' not in instruction_lower):
        # Extract zoom factor if specified, default to 1.5
        match = re.search(r'zoom.*?(\d+(?:\.\d+)?)', instruction_lower)
        zoom_factor = float(match.group(1)) if match else 1.5
        
        width, height = img.size
        new_width = int(width / zoom_factor)
        new_height = int(height / zoom_factor)
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        img = img.crop((left, top, right, bottom))
        img = img.resize((width, height), Image.LANCZOS)
        print(f"Zoomed in by factor {zoom_factor}")
    
    # **Zoom out functionality**
    if 'zoom out' in instruction_lower:
        match = re.search(r'zoom out.*?(\d+(?:\.\d+)?)', instruction_lower)
        zoom_factor = float(match.group(1)) if match else 0.7
        
        width, height = img.size
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Create new image with original size and paste resized image in center
        new_img = Image.new('RGB', (width, height), color='white')
        paste_x = (width - new_width) // 2
        paste_y = (height - new_height) // 2
        new_img.paste(img, (paste_x, paste_y))
        img = new_img
        print(f"Zoomed out by factor {zoom_factor}")
    
    # **Flip functionality**
    if 'flip horizontal' in instruction_lower or 'flip left' in instruction_lower:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        print("Flipped horizontally")
    elif 'flip vertical' in instruction_lower or 'flip up' in instruction_lower:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        print("Flipped vertically")
    
    # **Resize functionality**
    if 'resize' in instruction_lower:
        # Extract dimensions
        match = re.search(r'resize.*?(\d+).*?(\d+)', instruction_lower)
        if match:
            new_width, new_height = int(match.group(1)), int(match.group(2))
            img = img.resize((new_width, new_height), Image.LANCZOS)
            print(f"Resized to {new_width}x{new_height}")
    
    # **Blur functionality**
    if 'blur' in instruction_lower:
        match = re.search(r'blur.*?(\d+(?:\.\d+)?)', instruction_lower)
        radius = float(match.group(1)) if match else 2.0
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        print(f"Applied blur with radius {radius}")
    
    # **Brightness adjustment**
    if 'bright' in instruction_lower:
        match = re.search(r'bright.*?(\d+(?:\.\d+)?)', instruction_lower)
        factor = float(match.group(1)) if match else 1.3
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
        print(f"Adjusted brightness by factor {factor}")
    
    # **Contrast adjustment**
    if 'contrast' in instruction_lower:
        match = re.search(r'contrast.*?(\d+(?:\.\d+)?)', instruction_lower)
        factor = float(match.group(1)) if match else 1.2
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
        print(f"Adjusted contrast by factor {factor}")
    
    # **Convert to grayscale**
    if 'grayscale' in instruction_lower or 'gray' in instruction_lower:
        img = img.convert('L').convert('RGB')
        print("Converted to grayscale")
    
    # **Add border**
    if 'border' in instruction_lower:
        match = re.search(r'border.*?(\d+)', instruction_lower)
        border_width = int(match.group(1)) if match else 10
        
        # Extract color if specified
        color_match = re.search(r'border.*?(black|white|red|blue|green|yellow)', instruction_lower)
        border_color = color_match.group(1) if color_match else 'black'
        
        width, height = img.size
        new_img = Image.new('RGB', (width + 2*border_width, height + 2*border_width), color=border_color)
        new_img.paste(img, (border_width, border_width))
        img = new_img
        print(f"Added {border_color} border with width {border_width}")
    
    # Convert PIL Image back to base64 string
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_base64_output = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64_output

# Additional specialized tools
@mcp.tool()
def create_thumbnail(img_base64: str, size: int = 128) -> str:
    """Create a thumbnail from an image
    
    Args:
        img_base64: Input image as base64 encoded string
        size: Thumbnail size (default 128x128)
        
    Returns:
        str: Thumbnail as base64 encoded string
    """
    img_data = base64.b64decode(img_base64)
    img = Image.open(BytesIO(img_data))
    
    img.thumbnail((size, size), Image.LANCZOS)
    
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@mcp.tool()
def get_image_info(img_base64: str) -> dict:
    """Get information about an image
    
    Args:
        img_base64: Input image as base64 encoded string
        
    Returns:
        dict: Image information including size, format, mode
    """
    img_data = base64.b64decode(img_base64)
    img = Image.open(BytesIO(img_data))
    
    return {
        "width": img.size[0],
        "height": img.size[1],
        "mode": img.mode,
        "format": img.format or "Unknown"
    }

if __name__ == "__main__":
    print("\nStarting MCP Image Editing Service...")
    mcp.run(transport='stdio')
