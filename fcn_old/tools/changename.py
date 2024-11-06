import os

def rename_images(directory):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        if "_pred" in filename:
            # 构造新的文件名：去除 "_pred" 部分
            new_name = filename.replace("_pred", "")
            # 获取文件的完整路径
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_name)
            # 重命名文件
            os.rename(old_file, new_file)
            print(f"Renamed '{filename}' to '{new_name}'")

# 指定需要修改文件名的文件夹路径
directory_path = "D:\\data_set\\UECCOMPLETE\\Images\\predictions"
rename_images(directory_path)
