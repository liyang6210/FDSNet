# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
#
# def random_color():
#     # 生成随机颜色
#     return (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256), 204)
#
# # 加载原图和灰度图
# original_image = Image.open("C:\\Users\\liyang\\Desktop\\两个数据集对比图\\1726.jpg")
# mask_image = Image.open("C:\\Users\\liyang\\Desktop\\两个数据集对比图\\1726.png")
#
# # 确保灰度图是单通道
# if mask_image.mode != 'L':
#     mask_image = mask_image.convert('L')
#
# # 获取图像数据
# original_data = np.array(original_image)
# mask_data = np.array(mask_image)
#
# # 创建与原图同样大小的RGBA图像，用于覆盖
# overlay_image = np.zeros((original_data.shape[0], original_data.shape[1], 4), dtype=np.uint8)
#
# # 储存已经分配颜色的灰度值
# color_map = {}
#
# # 为每个非黑色的灰度值分配一个颜色，并应用到对应的位置
# for gray_value in np.unique(mask_data):
#     if gray_value != 0:  # 跳过背景
#         if gray_value not in color_map:
#             color_map[gray_value] = random_color()
#         overlay_image[mask_data == gray_value] = color_map[gray_value]
#
# # 将覆盖层转换为图像
# overlay_image_pil = Image.fromarray(overlay_image, 'RGBA')
#
# # 将原图转换为RGBA并与覆盖层合成
# original_rgba = original_image.convert('RGBA')
# combined_image = Image.alpha_composite(original_rgba, overlay_image_pil)
#
# # 保存或显示结果
# combined_image.save("path_to_save_combined_image.png")
# # combined_image.show()  # 如果需要查看图像，可以取消注释这行


import numpy as np
from PIL import Image

def apply_colored_overlay(original_image_path, mask_image_path, output_image_path, colors):
    # 加载原图和灰度图
    original_image = Image.open(original_image_path)
    mask_image = Image.open(mask_image_path)

    # 确保灰度图是单通道
    if mask_image.mode != 'L':
        mask_image = mask_image.convert('L')

    # 获取图像数据
    original_data = np.array(original_image)
    mask_data = np.array(mask_image)

    # 创建与原图同样大小的RGBA图像，用于覆盖
    overlay_image = np.zeros((original_data.shape[0], original_data.shape[1], 4), dtype=np.uint8)

    # 为每个独特的灰度值分配颜色并应用到对应的区域
    for gray_value, color in colors.items():
        rgba_color = (*color, 204)  # 解包RGB颜色并添加Alpha值
        overlay_image[mask_data == gray_value] = rgba_color

    # 将覆盖层转换为图像
    overlay_image_pil = Image.fromarray(overlay_image, 'RGBA')

    # 将原图转换为RGBA并与覆盖层合成
    original_rgba = original_image.convert('RGBA')
    combined_image = Image.alpha_composite(original_rgba, overlay_image_pil)

    # 保存结果
    combined_image.save(output_image_path)

# 使用示例
colors = {
    11: (222, 115, 94),

}
apply_colored_overlay("C:\\Users\\liyang\\Desktop\\两个数据集对比图\\UEC\\16786.jpg", "C:\\Users\\liyang\\Desktop\\两个数据集对比图\\UEC\\16786.png", "path_to_save_combined_image.png", colors)
