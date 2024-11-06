

import os
import cv2


def is_similar(imageA, imageB):
    # 将图像转换为灰度图
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # 使用模板匹配计算相似度
    result = cv2.matchTemplate(grayA, grayB, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)

    return max_val


def find_best_matching_image(folder_path, screenshot_path):
    screenshot = cv2.imread(screenshot_path)
    if screenshot is None:
        print(f"Failed to load screenshot: {screenshot_path}")
        return

    highest_score = 0
    best_match = None

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # 调整大小以匹配截图
        resized_img = cv2.resize(img, (screenshot.shape[1], screenshot.shape[0]))

        score = is_similar(resized_img, screenshot)
        if score > highest_score:
            highest_score = score
            best_match = img_name

    if best_match:
        print(f"Best matching image: {best_match} with score: {highest_score}")
    else:
        print("No matching image found.")


if __name__ == "__main__":
    folder_path = "D:\\data_set\\FoodSeg103\\Images\\img_dir\\test"  # 替换为你的图片文件夹路径
    screenshot_path = "../pic.png"  # 替换为你的截图路径

    find_best_matching_image(folder_path, screenshot_path)
