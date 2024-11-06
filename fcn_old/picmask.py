import cv2
import numpy as np

# def compute_and_save_laplacian(input_image_path, output_image_path):
#     # 读取输入图像
#     image = cv2.imread(input_image_path)
#     if image is None:
#         raise ValueError(f"无法读取图像文件 {input_image_path}")
#
#     # 转换为灰度图像
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # 计算拉普拉斯图像
#     laplacian_image = cv2.Laplacian(gray_image, cv2.CV_64F)
#
#     # 为了保存图像，将其转换为8位无符号整数
#     laplacian_image = cv2.convertScaleAbs(laplacian_image)
#
#     # 保存拉普拉斯图像
#     cv2.imwrite(output_image_path, laplacian_image)
#
# # 示例使用
# input_image_path = 'C:\\Users\\liyang\\Desktop\\Seg\\pic1.png'  # 输入图像文件路径
# output_image_path = './laplacian.jpg'  # 输出拉普拉斯图像文件路径
#
# compute_and_save_laplacian(input_image_path, output_image_path)


import cv2
import numpy as np

def compute_and_save_laplacian(input_image_path, output_image_path):
    # 读取输入图像
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"无法读取图像文件 {input_image_path}")

    # 初始化一个空的图像来存储每个通道的拉普拉斯图像
    laplacian_image = np.zeros_like(image)

    # 对每个颜色通道分别计算拉普拉斯图像
    for i in range(3):  # 对B, G, R三个通道
        channel = image[:, :, i]
        laplacian_channel = cv2.Laplacian(channel, cv2.CV_64F)
        laplacian_image[:, :, i] = cv2.convertScaleAbs(laplacian_channel)

    # 保存拉普拉斯图像
    cv2.imwrite(output_image_path, laplacian_image)

# 示例使用
input_image_path = 'C:\\Users\\liyang\\Desktop\\Seg\\pic1.png'  # 输入图像文件路径
output_image_path = './laplacian.jpg'  # 输出拉普拉斯图像文件路径

compute_and_save_laplacian(input_image_path, output_image_path)
