import os

def get_filenames_from_txt(txt_file):
    with open(txt_file, 'r') as file:
        filenames = file.read().splitlines()
    return filenames

def filter_images_in_folder(folder_path, txt_file):
    valid_filenames = get_filenames_from_txt(txt_file)
    images_in_folder = os.listdir(folder_path)
    matching_images = [img for img in images_in_folder if img in valid_filenames]

    for img in matching_images:
        print(img)

if __name__ == "__main__":
    folder_path = "D:\\data_set\\FoodSeg103\\Images\\Concat100"  # 替换为你的图片文件夹路径
    txt_file = "D:\\data_set\\FoodSeg103\\ImageSets\\test.txt"  # 替换为你的txt文件路径

    filter_images_in_folder(folder_path, txt_file)
