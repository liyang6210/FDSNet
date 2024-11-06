from PIL import Image
import os

def overlay_images(source_folder, target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的文件
    for file in os.listdir(source_folder):
        if file.endswith(".png"):
            base_name = os.path.splitext(file)[0]
            # 找到对应的JPG文件
            jpg_file = os.path.join(source_folder, base_name + ".jpg")
            png_file = os.path.join(source_folder, base_name + ".png")

            if os.path.exists(jpg_file):
                # 打开图片
                img_jpg = Image.open(jpg_file).convert("RGBA")
                img_png = Image.open(png_file).convert("RGBA")

                # 调整PNG图片的透明度
                img_png.putalpha(160)  # 0.8 * 255 ≈ 204

                # 合成图片
                result_img = Image.alpha_composite(img_jpg, img_png)

                # 保存结果
                result_img.save(os.path.join(target_folder, base_name + "_overlay.png"))


# 使用示例
source_folder = 'C:\\Users\\liyang\\Desktop\\结果对比图\\MyNet\\103'
target_folder = 'C:\\Users\\liyang\\Desktop\\结果对比图\\MyNet\\处理之后的\\SegFormer640'
overlay_images(source_folder, target_folder)
