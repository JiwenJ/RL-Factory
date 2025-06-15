from PIL import Image, ImageDraw
from mcp.server.fastmcp import FastMCP
from io import BytesIO
import base64

# Initialize MCP server
mcp = FastMCP("ImageEditServer")

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


    



