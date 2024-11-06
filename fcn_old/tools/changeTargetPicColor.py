# #103
#
#
#
# import os
# import json
# from PIL import Image
#
# # 加载映射表，将灰度值映射到RGB值
# def load_mapping(json_file):
#     with open(json_file, 'r') as file:
#         mapping = json.load(file)
#     return {int(k): tuple(v) for k, v in mapping.items()}
#
# # 处理单张图片，将灰度值替换为对应的RGB值
# def process_image(image_path, mapping, output_folder):
#     # 打开灰度图像
#     image = Image.open(image_path)
#     pixels = image.load()
#
#     # 创建一个新的RGB图像
#     color_image = Image.new("RGB", image.size)
#     color_pixels = color_image.load()
#
#     # 替换灰度值
#     for x in range(image.width):
#         for y in range(image.height):
#             gray = pixels[x, y]
#             if gray in mapping:
#                 color_pixels[x, y] = mapping[gray]
#             else:
#                 color_pixels[x, y] = (0, 0, 0)  # 未定义的灰度值映射为黑色
#
#     # 保存新图像
#     base_name = os.path.basename(image_path)
#     color_image.save(os.path.join(output_folder, base_name))
#
# # 主函数
# def main(input_folder, json_file, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     mapping = load_mapping(json_file)
#     for file_name in os.listdir(input_folder):
#         file_path = os.path.join(input_folder, file_name)
#         if file_path.endswith('.jpg') or file_path.endswith('.png'):
#             process_image(file_path, mapping, output_folder)
#
# # 示例用法
# input_folder = 'D:\\data_set\\FoodSeg103\\Images\\ann_dir\\test'
# json_file = '../palette.json'
# output_folder = 'D:\\data_set\\FoodSeg103\\Images\\ann_dir\\colortest'
# main(input_folder, json_file, output_folder)



# UEC
import os
import json
from PIL import Image

# 加载映射表，将R通道值映射到RGB值
def load_mapping(json_file):
    with open(json_file, 'r') as file:
        mapping = json.load(file)
    return {int(k): tuple(v) for k, v in mapping.items()}

# 处理单张图片，使用R通道值作为键名映射到新的RGB值
def process_image(image_path, mapping, output_folder):
    # 打开图像
    image = Image.open(image_path)
    pixels = image.load()

    # 创建一个新的RGB图像
    color_image = Image.new("RGB", image.size)
    color_pixels = color_image.load()

    # 替换R通道值
    for x in range(image.width):
        for y in range(image.height):
            r_value = pixels[x, y][0]  # 获取R通道值
            if r_value in mapping:
                color_pixels[x, y] = mapping[r_value]
            else:
                color_pixels[x, y] = (0, 0, 0)  # 未定义的R通道值映射为黑色

    # 保存新图像
    base_name = os.path.basename(image_path)
    color_image.save(os.path.join(output_folder, base_name))

# 主函数
def main(input_folder, json_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    mapping = load_mapping(json_file)
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if file_path.endswith('.jpg') or file_path.endswith('.png'):
            process_image(file_path, mapping, output_folder)

# 示例用法
input_folder = 'D:\\data_set\\UECFOODPIXCOMPLETE\\data\\UECFoodPIXCOMPLETE\\test\\mask'
json_file = '../palette.json'
output_folder = 'D:\\data_set\\UECFOODPIXCOMPLETE\\data\\UECFoodPIXCOMPLETE\\test\\colormask'
main(input_folder, json_file, output_folder)
