from PIL import Image, ImageDraw
from mcp.server.fastmcp import FastMCP
from io import BytesIO
import base64

# Initialize MCP server
mcp = FastMCP("ImageEditServer")



@mcp.tool()
def edit_image_pil(img_base64: str, instruction: str) -> str:
    """Edit a Pillow image based on natural language instructions
    
    Args:
        img_base64: Input image as base64 encoded string
        instruction: Natural language editing instructions
        
    Returns:
        str: Edited image as base64 encoded string
    """
    # Convert base64 string to PIL Image
    img_data = base64.b64decode(img_base64)
    img = Image.open(BytesIO(img_data))
    
    # First handle cropping if specified
    if "crop" in instruction.lower():
        if "center" in instruction:
            img = img.crop((img.width//4, img.height//4, 
                            img.width*3//4, img.height*3//4))
        elif "square" in instruction:
            size = min(img.width, img.height)
            img = img.crop(((img.width - size) // 2,
                            (img.height - size) // 2,
                            (img.width + size) // 2,
                            (img.height + size) // 2))
        elif "region" in instruction:
            # Extract coordinates from instruction
            coords = []
            parts = instruction.split()
            for part in parts:
                if ',' in part:
                    try:
                        coords = tuple(map(int, part.split(',')))
                        if len(coords) == 4:
                            img = img.crop(coords)
                            break
                    except ValueError:
                        continue
    
    # Then handle resizing if specified
    if "resize" in instruction.lower():
        # Use high-quality LANCZOS resampling
        resample = Image.LANCZOS
        
        if "50%" in instruction:
            new_width = img.width // 2
            new_height = img.height // 2
            img = img.resize((new_width, new_height), resample)
        elif "double" in instruction:
            new_width = img.width * 2
            new_height = img.height * 2
            img = img.resize((new_width, new_height), resample)
        elif "thumbnail" in instruction:
            img.thumbnail((128, 128), resample)
        elif "width" in instruction:
            # Extract target width
            parts = instruction.split()
            for part in parts:
                if part.isdigit():
                    width = int(part)
                    ratio = width / img.width
                    height = int(img.height * ratio)
                    img = img.resize((width, height), resample)
                    break
        elif "to" in instruction:
            # Handle "resize to 800x600" format with aspect ratio preservation
            parts = instruction.split()
            for i, part in enumerate(parts):
                if part == "to" and i+1 < len(parts):
                    dim_str = parts[i+1]
                    if 'x' in dim_str:
                        try:
                            # Get target dimensions
                            target_width, target_height = map(int, dim_str.split('x'))
                            
                            # Calculate dimensions preserving aspect ratio
                            width_ratio = target_width / img.width
                            height_ratio = target_height / img.height
                            
                            # Use the more constraining ratio
                            ratio = min(width_ratio, height_ratio)
                            
                            new_width = int(img.width * ratio)
                            new_height = int(img.height * ratio)
                            
                            img = img.resize((new_width, new_height), resample)
                            break
                        except (ValueError, TypeError):
                            # Fallback to original if error
                            continue
        elif "maintain ratio" in instruction.lower():
            # Extract scale factor
            parts = instruction.split()
            for part in parts:
                try:
                    # Handle "200%" format
                    if '%' in part:
                        scale = float(part.strip('%')) / 100
                        new_width = int(img.width * scale)
                        new_height = int(img.height * scale)
                        img = img.resize((new_width, new_height), resample)
                        break
                    # Handle "2.0" format
                    elif '.' in part:
                        scale = float(part)
                        new_width = int(img.width * scale)
                        new_height = int(img.height * scale)
                        img = img.resize((new_width, new_height), resample)
                        break
                except ValueError:
                    continue
    
    # Handle other operations (rotation, color, etc.)
    if "rotate" in instruction.lower():
        if "90" in instruction:
            img = img.rotate(90, expand=True)
        elif "180" in instruction:
            img = img.rotate(180)
        elif "270" in instruction:
            img = img.rotate(270, expand=True)
            
    if "grayscale" in instruction.lower() or "black and white" in instruction.lower():
        img = img.convert("L")
    elif "sepia" in instruction.lower():
        # Apply sepia tone
        sepia = img.convert("RGB")
        width, height = sepia.size
        pixels = sepia.load()
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                pixels[x, y] = (min(255, tr), min(255, tg), min(255, tb))
        img = sepia
        
    if "flip" in instruction.lower():
        if "horizontal" in instruction:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif "vertical" in instruction:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
    if "bright" in instruction.lower():
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.5)
    elif "dark" in instruction.lower():
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.7)
            
    # Convert the edited image back to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64_out = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64_out

if __name__ == "__main__":
    print("\nStarting MCP Image Editing Service...")
    mcp.run(transport='stdio')


    



