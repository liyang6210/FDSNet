import os
import time
import math
import datetime

import torch
# from torchvision import transforms as T
# from src import Seg_Model
# from src import SETR
from src import SegFormer
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import VOCSegmentation
import transforms as T
import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]  # 将图像的最小边变成一个在min_size和max_size中间的尺寸
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):

        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.RandomCrop(base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train):
    base_size = 768
    crop_size = 768

    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)


def create_model(aux, num_classes, pretrain=False):
    model = SegFormer(num_classes=num_classes)

    if pretrain:
        weights_dict = torch.load("D:/data_set/weight_ported/resnet101-imagenet.pth", map_location='cpu')
        if num_classes == 104:
            new_weight_dict = {}

            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                weights_dict['resnet101.' + k] = weights_dict[k]
                del weights_dict[k]
                # print(weights_dict[k])
                # print("k1",k)
                # k = "resnet101." + k
                # print("k2",k)
                if "classifier.4" in k:
                    del weights_dict[k]
        print(weights_dict.keys())

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)  # 网络中有、需要的，但是权重数据中没有这些数据，也就是网络中缺少的
            print("unexpected_keys: ", unexpected_keys)  # 网络中没有这些，但是权重数据中有这些，网络不需要的

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
    train_dataset = VOCSegmentation(args.data_path,
                                    transforms=get_transform(train=True),
                                    txt_name="train.txt")

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    test_dataset = VOCSegmentation(args.data_path,
                                   transforms=get_transform(train=False),
                                   txt_name="test.txt")

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              collate_fn=test_dataset.collate_fn)

    model = create_model(aux=args.aux, num_classes=num_classes)
    model.to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        # del_keys = ['head.weight', 'head.bias'] if model.has_logits \
        #     else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        del_keys = ['decode_head.linear_pred.weight', 'decode_head.linear_pred.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    params_to_optimize = [
        {"params": [p for p in model.parameters() if p.requires_grad]},
        # {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    if args.aux:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        # lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        lr=args.lr, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    lr_scheduler.ExponentialLR
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=scheduler, print_freq=args.print_freq, scaler=scaler)
        if epoch % 2 == 0 and epoch != 0:
            confmat = evaluate(model, test_loader, device=device, num_classes=num_classes)
            print("confmat", confmat)
            val_info = str(confmat)
            print(val_info)
            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n"
                f.write(train_info + val_info + "\n\n")

            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
            if epoch % 4 == 0:
                torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("--data-path", default="D:/data_set")
    parser.add_argument("--num-classes", default=103, type=int)
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument("--epochs", default=720, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--weights', type=str, default='D:\data_set\weight_ported\segformer_b2_weights_voc.pth',
                        help='initial weights path')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
