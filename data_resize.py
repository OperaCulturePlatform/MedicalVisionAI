from PIL import Image, ImageOps, ImageDraw
import os
import re

def crop_and_resize_image(input_path, output_path):
    # 打开输入图像
    image = Image.open(input_path).convert("RGBA")
    
    # 创建一个透明度掩码
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    
    # 获取图像的宽度和高度
    width, height = image.size
    
    # 遍历每个像素点
    for x in range(width):
        for y in range(height):
            r, g, b, a = image.getpixel((x, y))
            if (r, g, b) != (0, 0, 0):  # 如果不是黑色，则设置掩码为白色
                draw.point((x, y), fill=255)
    
    # 使用掩码切割图像
    bounding_box = mask.getbbox()
    if not bounding_box:
        return
    
    cropped_image = image.crop(bounding_box)
    
    # 缩放图像到半径为192的圆形
    eye_radius = 192
    resized_image = cropped_image.resize((eye_radius * 2, eye_radius * 2), Image.Resampling.LANCZOS)

    # 创建一个圆形蒙版
    circle_mask = Image.new('L', (eye_radius * 2, eye_radius * 2), 0)
    draw_circle = ImageDraw.Draw(circle_mask)
    draw_circle.ellipse((0, 0, eye_radius * 2, eye_radius * 2), fill=255)
    
    # 将圆形蒙版应用到缩放后的图像
    masked_image = ImageOps.fit(resized_image, (eye_radius * 2, eye_radius * 2), centering=(0.5, 0.5))
    masked_image.putalpha(circle_mask)
    
    # 创建一个新的透明背景图像
    new_size = (384, 384)
    background = Image.new('RGBA', new_size, (0, 0, 0, 0))
    
    # 将缩放后的图像粘贴到背景上
    paste_position = ((new_size[0] - eye_radius * 2) // 2, (new_size[1] - eye_radius * 2) // 2)
    background.paste(masked_image, paste_position, masked_image.split()[3])
    
    # 保存最终图像
    background.save(output_path, format='PNG')

def natural_sort_key(s):
    """Sorts strings with embedded numbers in a natural way."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def process_images_in_folder(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有文件并按自然顺序排序
    files = sorted(os.listdir(input_folder), key=natural_sort_key)
    
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            crop_and_resize_image(input_path, output_path)
            print(f"Processed {filename}")


# 示例用法
if __name__ == "__main__":
    input_folder = r"D:\ComputerTool\tmp\train"
    output_folder = r"D:\ComputerTool\tmp\train1"
    process_images_in_folder(input_folder, output_folder)



