"""
为LIP数据集生成edge标签
"""

import os
import sys
import numpy as np
import random
import cv2
from tqdm import tqdm

def generate_edge(label, edge_width=1):
    h, w = label.shape
    edge = np.zeros(label.shape)

    # right
    edge_right = edge[1:h, :]
    edge_right[(label[1:h, :] != label[:h - 1, :]) & (label[1:h, :] != 255)
               & (label[:h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :w - 1]
    edge_up[(label[:, :w - 1] != label[:, 1:w])
            & (label[:, :w - 1] != 255)
            & (label[:, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:h - 1, :w - 1]
    edge_upright[(label[:h - 1, :w - 1] != label[1:h, 1:w])
                 & (label[:h - 1, :w - 1] != 255)
                 & (label[1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:h - 1, 1:w]
    edge_bottomright[(label[:h - 1, 1:w] != label[1:h, :w - 1])
                     & (label[:h - 1, 1:w] != 255)
                     & (label[1:h, :w - 1] != 255)] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
    edge = cv2.dilate(edge, kernel)
    return edge
  
def _box2cs(box, aspect_ratio, pixel_std):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, aspect_ratio, pixel_std)


def _xywh2cs(x, y, w, h, aspect_ratio, pixel_std):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    # if center[0] != -1:
    #    scale = scale * 1.25

    return center, scale

if __name__ == '__main__':
    process_unit = 'folder'
    if process_unit == 'image':
        img_path = input('Input image filename:')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        edge = generate_edge(img)
        cv2.imwrite(img_path.split('/')[-1], edge)
    elif process_unit == 'folder':
        root = '../data/LIP_2'
        folder_list = ["val"]
        for folder in folder_list:
            print("=========================正在处理", folder, "=================================")
            imgs = os.listdir(os.path.join(root, folder, "gt"))

            for img_name in tqdm(imgs):
                if img_name.lower().endswith(('.png', '.jpg')):
                    img = cv2.imread(os.path.join(root, folder, "gt", img_name), cv2.IMREAD_GRAYSCALE)
                    edge = generate_edge(img)
                    save_root = os.path.join(root, folder, "gt_edge")
                    if not os.path.exists(save_root):
                        os.makedirs(save_root)
                    cv2.imwrite(os.path.join(save_root, img_name), edge)
                else:
                    # print(img_name, "is not a image!")
                    continue

