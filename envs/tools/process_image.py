import PIL.Image

image_file = "/root/autodl-tmp/RL-Factory/envs/tools/table1.jpg"
image = PIL.Image.open(image_file)
print("original image size:", image.size)
from qwen_vl_utils import smart_resize
resized_image = smart_resize(image.size[0], image.size[1], max_pixels=1024 * 28 * 28)
print("resized image size:", resized_image)
new_image = image.resize(resized_image)
new_image.save(image_file.replace(".png", "_400.png"))
