import os
import numpy as np
from PIL import Image

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def main():
    predictions_folder = "D:\\data_set\\UECFOODPIXCOMPLETE\\data\\UECFoodPIXCOMPLETE\\test\\predict"
    targets_folder = "D:\\data_set\\UECFOODPIXCOMPLETE\\data\\UECFoodPIXCOMPLETE\\test\\mask"
    output_folder = "D:\\data_set\\UECFOODPIXCOMPLETE\\data\\UECFoodPIXCOMPLETE\\test\\100IOU"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    iou_scores = []

    for file_name in os.listdir(predictions_folder):
        pred_path = os.path.join(predictions_folder, file_name)
        target_path = os.path.join(targets_folder, file_name)

        if os.path.exists(target_path):
            pred_img = Image.open(pred_path)
            target_img = Image.open(target_path)

            pred_arr = np.array(pred_img)
            target_arr = np.array(target_img)

            # Convert to binary masks where non-background is True
            pred_mask = np.any(pred_arr != [0, 0, 0], axis=-1)
            target_mask = target_arr > 0  # Assuming background is labeled as 0

            iou_score = calculate_iou(pred_mask, target_mask)
            iou_scores.append((file_name, iou_score))

    # Sort by IoU score and select the top 100
    top_files = sorted(iou_scores, key=lambda x: x[1], reverse=True)[:100]

    # Copy the top 100 prediction images to a new folder
    for file_name, _ in top_files:
        source_path = os.path.join(predictions_folder, file_name)
        destination_path = os.path.join(output_folder, file_name)
        os.rename(source_path, destination_path)

    print(f"Top 100 images with highest IoU moved to {output_folder}")

if __name__ == '__main__':
    main()
