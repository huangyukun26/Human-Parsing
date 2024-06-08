"""
对ReID数据集进行预处理
运行前需修改：
1.__process_unit: image, folder, dataset
    if image: save_root
    if folder: folder_list, save_root
    if dataset: dataset, save_root
2.__save_mode: 1,2,3,4
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import logging
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from copy import deepcopy
from torchvision import transforms
from networks.CE2P_PSD2_DA_new import Res_Deeplab

parser = argparse.ArgumentParser(description="Pre-processing of ReID dataset segmentation")
# parser.add_argument('--dataset', type=str, default='data/market1501', help='Datasets that require segmentation')
parser.add_argument('--process_unit', type=str, default='folder', help='processing units: image, folder, and dataset')
parser.add_argument('--save_mode', type=int, default=1, help='Visualization of segmentation results')
parser.add_argument('--restore-from', type=str, default='snapshots_LIP2/ce2p_psd2_da_bs8_2/LIP_epoch_149.pth', help='Path for storing model snapshots')
parser.add_argument('--num_classes', type=int, default=2, help="Number of classes")
args = parser.parse_args()

def segment(ori_img):
    # --------------- model --------------- #
    model = Res_Deeplab(num_classes=args.num_classes)

    restore_from = args.restore_from

    state_dict = model.state_dict().copy()
    state_dict_old = torch.load(restore_from)

    for key, nkey in zip(state_dict_old.keys(), state_dict.keys()):
        if key != nkey:
            # remove the 'module.' in the 'key'
            state_dict[key[7:]] = deepcopy(state_dict_old[key])
        else:
            state_dict[key] = deepcopy(state_dict_old[key])

    model.load_state_dict(state_dict)

    model.eval()
    model.cuda()

    # ------------ load image ------------ #
    data_transform = transforms.Compose([
        transforms.Resize((256, 128), 3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = data_transform(ori_img).cuda()

    # --------------- inference --------------- #
    with torch.no_grad():
        outputs = model(img.unsqueeze(dim=0))
        pred = outputs[0][-1]
        pred = pred.squeeze(dim=0).cpu().numpy().transpose(1, 2, 0)
        pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8).reshape((256, 128, 1))
        h, w, _ = pred.shape
        pred = pred.reshape((h, w))
    return pred

def save_img(ori_img, pred, path):  # 传过来的pred是一个np数组
    # pred:[256,128]
    # -------------------------------------------------#
    #   save_mode参数用于控制检测结果的可视化方式
    #
    #   save_mode = 0  原图与生成的图进行混合
    #   save_mode = 1  直接保留二值预测图，结果保留在market1501_binary中
    #   save_mode = 2  去除背景，仅保留行人信息，结果保留在market1501_binary_foreground中
    #   save_mode = 3  去除前景，仅保留背景信息，结果保留在market1501_binary_background中
    #   save_mode = 4  将背景随机在[0,255]之间取一个值，前景不变，结果存在 market1501_random_background中
    # -------------------------------------------------#

    w, h = ori_img.size  # w=64, h=128
    if args.save_mode == 1:
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        pred = np.where(pred == 0, 0, 255)  # 为了便于查看，将灰度值为1的位置改为255
        pred = Image.fromarray(np.uint8(pred)).convert('L')

        pred.save(path)

    elif args.save_mode == 2:
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        pred.argmax(axis=-1)
        foreground = (np.expand_dims(pred != 0, axis=-1) * np.array(ori_img, np.float32)).astype('uint8')  # fore_img:[256, 128, 3]
        foreground = Image.fromarray(np.uint8(foreground))
        foreground.save(path)
    elif args.save_mode == 3:
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        pred.argmax(axis=-1)
        background = (np.expand_dims(pred == 0, axis=-1) * np.array(ori_img, np.float32)).astype('uint8')  # fore_img:[128, 64, 3]
        background = Image.fromarray(np.uint8(background))
        background.save(path)

    elif args.save_mode == 4:
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        pred.argmax(axis=-1)
        foreground = (np.expand_dims(pred != 0, axis=-1) * np.array(ori_img, np.float32)).astype('uint8')  # fore_img:[128, 64, 3]
        # background = (np.expand_dims(pred == 0, axis=-1) * np.array(ori_img, np.float32)).astype('uint8')
        for num1 in foreground:
            for num2 in num1:
                for index, num3 in enumerate(num2):
                    if num3 == 0:
                        num2[index] = np.random.randint(0, 255)
        random_back = Image.fromarray(np.uint8(foreground))
        random_back.save(path)

if __name__ == '__main__':

    # ----------------------------------------------------------------------------------------------------------#
    #   process_unit用于指定测试的模式：
    #   'image'     表示对单张图片进行分割
    #   'folder'    表示对一个文件夹中的所有图片进行分割
    #   ‘dataset’   表示对一个数据集中的所有图片进行分割
    # ----------------------------------------------------------------------------------------------------------#
    if args.process_unit == 'image':
        img_path = input('Input image filename:')
        img = Image.open(img_path)
        pred = segment(img)
        save_root = 'SegmentProcess' + img_path.split('/')[-1]
        save_img(img, pred, save_root)

    elif args.process_unit == 'folder':
        # root = 'data/market1501'
        # folder_list = ["bounding_box_train", "bounding_box_test", "query"]
        root = ''
        folder_list = ["demo"]
        for folder in folder_list:
            print("=========================正在处理", folder, "=================================")
            imgs = os.listdir(os.path.join(root, folder))
            for img_name in tqdm(imgs):
                if img_name.lower().endswith(('.png', '.jpg')):
                    img = Image.open(os.path.join(root, folder, img_name))
                    pred = segment(img)
                    save_root = os.path.join(root, folder+'_mask_new')
                    # print(save_root)
                    if not os.path.exists(save_root):
                        os.makedirs(save_root)
                    save_img(img, pred, os.path.join(save_root, img_name))
                else:
                    # print(img_name, "is not a image!")
                    continue

    # elif args.process_unit == 'dataset':
    #     dataset = 'market1501'
    #     save_root = "market1501_binary"
    #     files = os.listdir(args.dataset)
    #     print("files in origin_root:", files)
    #
    #     for file in files:
    #         file_path = os.path.join(args.dataset, file)  # 例如：data/market1501/bounding_box_test
    #         if os.path.isdir(file_path):
    #             print("=========================正在处理", file, "=================================")
    #             save_path = os.path.join(save_root, file)
    #             if not os.path.exists(save_path):
    #                 os.makedirs(save_path)
    #
    #             img_names = os.listdir(file_path)
    #             for img_name in tqdm(img_names):
    #                 if img_name.lower().endswith(('.png', '.jpg')):
    #                     # print(index, ": ", img_name)
    #                     image_path = os.path.join(file_path, img_name)
    #                     img = Image.open(image_path)
    #                     pred = segment(img)
    #                     save_img(img, pred, os.path.join(save_path, img_name))
    #                 else:
    #                     print(img_name, "is not a image!")
    #                     continue
    #         else:
    #             print(file, "is not a dir!")
    #             continue