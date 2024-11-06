import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.nn import functional as F
from src import SegFormer


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():

    weights_path = "D:\\study\\实验记录\\SegFomer\\SegFormer42.5.pth"
    input_folder = "C:\\Users\\liyang\\Desktop\\结果对比图\\MyNet\\103"
    output_folder = "C:\\Users\\liyang\\Desktop\\结果对比图\\MyNet\\处理之后的\\SegFormer640"
    palette_path = "../palette.json"

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(input_folder), f"input folder {input_folder} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(palette_path, "rb") as f:
        palette_dict = json.load(f)
        palette = []
        for v in palette_dict.values():
            palette += v

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = SegFormer(num_classes=104)
    weights_dict = torch.load(weights_path, map_location='cpu')
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]
    model.load_state_dict(weights_dict['model'])
    model.to(device)

    data_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    model.eval()
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        original_img = Image.open(img_path)
        img = data_transform(original_img)
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            output = model(img.to(device))
            output = F.interpolate(output, size=(original_img.height, original_img.width), mode='bilinear',
                                   align_corners=False)
            prediction = output.argmax(1).squeeze(0).to("cpu").numpy().astype(np.uint8)
            mask = Image.fromarray(prediction)
            mask.putpalette(palette)

            result_path = os.path.join(output_folder, os.path.splitext(img_name)[0] + ".png")
            mask.save(result_path)


if __name__ == '__main__':
    main()
