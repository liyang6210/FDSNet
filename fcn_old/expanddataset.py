from PIL import Image

# 图像文件路径
image_path = 'D:\\data_set\\FoodSeg103Big\\Images\\ann_dir\\all\\00008493.png'  # 替换为你的图像文件路径

# 打开图像文件
image = Image.open(image_path)

# 确保图像是PNG格式
if image.format != 'PNG':
    print("这不是PNG图像。")
else:
    # 获取图像的宽度和高度
    width, height = image.size

    # 遍历图像中的每个像素
    for x in range(width):
        for y in range(height):
            # 获取当前像素的RGB值
            pixel = image.getpixel((x, y))

            # 打印当前像素的RGB值
            print(f"Pixel at ({x}, {y}): RGB - {pixel}")