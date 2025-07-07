#!/usr/bin/env python3
"""
测试远程图片显示功能的示例脚本
"""

import os
import sys
from remote_image_display import RemoteImageDisplay, quick_show, quick_show_multiple, create_web_gallery

def main():
    print("=== 远程服务器图片显示测试 ===\n")
    
    # 获取当前目录下的图片文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_files = []
    
    # 查找图片文件
    for file in os.listdir(current_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_files.append(os.path.join(current_dir, file))
    
    if not image_files:
        print("当前目录下没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片:")
    for img in image_files:
        print(f"  - {os.path.basename(img)}")
    print()
    
    # 创建显示工具实例
    display = RemoteImageDisplay(output_dir="./image_outputs")
    
    # 方法1: 使用matplotlib显示单张图片
    print("=== 方法1: 使用matplotlib显示单张图片 ===")
    first_image = image_files[0]
    display.print_image_info(first_image)
    saved_path = display.show_with_matplotlib(first_image, f"显示图片: {os.path.basename(first_image)}")
    print(f"处理后的图片保存在: {saved_path}\n")
    
    # 方法2: 显示多张图片
    if len(image_files) > 1:
        print("=== 方法2: 显示多张图片 ===")
        titles = [f"图片 {i+1}: {os.path.basename(img)}" for i, img in enumerate(image_files)]
        saved_path = display.show_multiple_images(image_files[:4], titles[:4])  # 最多显示4张
        print(f"多张图片拼接保存在: {saved_path}\n")
    
    # 方法3: 创建HTML画廊
    print("=== 方法3: 创建HTML网页画廊 ===")
    titles = [os.path.basename(img) for img in image_files]
    html_path = display.create_html_display(image_files, titles)
    print(f"HTML画廊创建完成: {html_path}")
    print("你可以通过以下方式查看:")
    print("1. 直接用浏览器打开HTML文件")
    print("2. 启动本地HTTP服务器:")
    print(f"   cd {display.output_dir}")
    print("   python -m http.server 8000")
    print("   然后在浏览器中访问: http://localhost:8000/image_display.html\n")
    
    # 方法4: 使用便捷函数
    print("=== 方法4: 使用便捷函数 ===")
    quick_show(first_image, "快速显示测试")
    
    if len(image_files) > 1:
        quick_show_multiple(image_files[:3], ["图片A", "图片B", "图片C"])
    
    print("\n=== 所有测试完成 ===")
    print(f"输出文件保存在目录: {display.output_dir}")
    print("你可以使用以下命令查看生成的文件:")
    print(f"ls -la {display.output_dir}")

if __name__ == "__main__":
    main() 