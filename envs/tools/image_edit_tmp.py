from PIL import Image, ImageDraw
# from mcp.server.fastmcp imp,ort FastMCP

# mcp = FastMCP("LocalServer")

def focus_on_x_values_with_mask(image, x_values_to_focus_on, all_x_values_bounding_boxes):
    """
    This function is useful when you want to focus on some specific x values in the image.
    It does this by masking out the x values that are not needed.
    This function is especially useful for vertical bar charts.
    For example, you can focus on the x values in a chart that are relevant to your analysis and ignore the rest.
    Return the masked image.

    Args:
        image (PIL.Image.Image): the input image
        x_values_to_focus_on (List[str]): a list of x values to focus on. 
        all_x_values_bounding_boxes (Dict[Dict]): a dictionary of bounding boxes for all x values in the image. key is x value and value is the bounding box of that x value. Each bounding box is in the format {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}.
    
    Returns:
        image_with_focused_x_values (PIL.Image.Image): the image with specified x values focused on

    Example:
        image = Image.open("sample_img.jpg")
        image_with_focused_x_values = focus_on_x_values(image, ["2005", "2006"], {"2005": {'x1': 0.1, 'y1': 0.1, 'x2': 0.3, 'y2': 0.9}, "2006": {'x1': 0.4, 'y1': 0.1, 'x2': 0.6, 'y2': 0.9}, "2007": {'x1': 0.7, 'y1': 0.1, 'x2': 0.9, 'y2': 0.9}})
        display(image_with_focused_x_values)
    """
    # Convert image to RGBA if it's not already
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    # Create a drawing context for the image
    draw = ImageDraw.Draw(image, "RGBA")

    # Desipte the x values to focus on, mask out all other x values
    x_values_to_mask = [x_value for x_value in all_x_values_bounding_boxes if x_value not in x_values_to_focus_on]
    if len(x_values_to_mask) == len(all_x_values_bounding_boxes):
        return image
    
    # Iterate over the x values to mask out

    draw.rectangle(((100, 100), (120, 120)), fill="black")
    
    return image


# @mcp.tool()
def edit_image_pil(img: Image.Image, instruction: str) -> Image.Image:
    """Edit a Pillow image based on natural language instructions
    
    Args:
        img: Input PIL.Image.Image object
        instruction: Natural language editing instructions
        
    Returns:
        Image: Edited PIL.Image.Image object
    """
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
            
    return img

if __name__ == "__main__":
    input_path = "/Users/jiwen/Github/RL-Factory/envs/tools/bigmap.jpg"
    output_path = "/Users/jiwen/Github/RL-Factory/envs/tools/bigmap_processed.jpg"
    img = Image.open(input_path)
    
    # Get image dimensions
    width, height = img.size
    print(f"Original size: {width}x{height}")
    
    # Single instruction for both operations
    processed_img = edit_image_pil(
        img,
        instruction=(
            "crop region 0,{top},{right},{bottom} "  # Bottom-left region
            "and resize to 2000x1500"                 # High-res output
        ).format(
            top=height // 2,
            right=width // 2,
            bottom=height
        )
    )
    
    # Save with high quality
    processed_img.save(output_path, quality=95)
    print(f"Processed image saved to: {output_path}")


    



