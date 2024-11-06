import os
from PIL import Image

def main():
    folder_a = "D:\\data_set\\UECFOODPIXCOMPLETE\\data\\UECFoodPIXCOMPLETE\\test\\100IOU"  # A文件夹的路径
    folder_b = "D:\\data_set\\UECFOODPIXCOMPLETE\\data\\UECFoodPIXCOMPLETE\\test\\colormask"  # B文件夹的路径
    output_folder = "D:\\data_set\\UECFOODPIXCOMPLETE\\data\\UECFoodPIXCOMPLETE\\test\\100Concat"  # 输出文件夹的路径

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_a):
        img_a_path = os.path.join(folder_a, filename)
        img_b_path = os.path.join(folder_b, filename)

        if os.path.exists(img_b_path):
            img_a = Image.open(img_a_path)
            img_b = Image.open(img_b_path)

            # 确保两张图片的尺寸相同
            if img_a.size != img_b.size:
                print(f"Warning: Size mismatch for {filename}, resizing images...")
                img_b = img_b.resize(img_a.size)

            # 拼接图片
            total_width = img_a.width + img_b.width
            total_height = max(img_a.height, img_b.height)
            new_img = Image.new('RGB', (total_width, total_height))

            new_img.paste(img_a, (0, 0))
            new_img.paste(img_b, (img_a.width, 0))

            # 保存新图片
            new_img.save(os.path.join(output_folder, filename))

    print(f"Images successfully concatenated and saved in {output_folder}")

if __name__ == '__main__':
    main()


