import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
from networks.CE2P_PSD2_DA_new import Res_Deeplab
from dataset.datasets import LIPDataSet
from utils.transforms import transform_parsing
import os
import torchvision.transforms as transforms
from utils.miou import compute_mean_ioU
from copy import deepcopy
from dataset import target_generation

# def get_arguments():
#     """Parse all the arguments provided from the CLI.
#
#     Returns:
#       A list of parsed arguments.
#     """
parser = argparse.ArgumentParser(description="CE2P Network")
parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
parser.add_argument("--restore-from", type=str, default='./snapshots_LIP2/ce2p_psd2_da_bs8_2/LIP_epoch_149.pth', help="Where restore model parameters from.")
parser.add_argument("--input-size", type=str, default='256,128', help="Comma-separated string with height and width of images.")
parser.add_argument("--batch-size", type=int, default=1, help="Number of images sent to the network in one step.")
parser.add_argument("--data-dir", type=str, default='data/LIP_2', help="Path to the directory containing the PASCAL VOC dataset.")
parser.add_argument("--dataset", type=str, default='val', help="Path to the file listing the images in the dataset.")
parser.add_argument("--ignore-label", type=int, default=255, help="The index of the label to ignore during the training.")
parser.add_argument("--num-classes", type=int, default=2, help="Number of classes to predict (including background).")
args = parser.parse_args()

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def cam_mask(mask, colormap, num_classes):
    result = np.zeros((np.shape(mask)[0], np.shape(mask)[1], 3))

    for n in range(num_classes):
        result[:, :, 0] += ((mask[:, :] == n) * (colormap[n][0])).astype('uint8')
        result[:, :, 1] += ((mask[:, :] == n) * (colormap[n][1])).astype('uint8')
        result[:, :, 2] += ((mask[:, :] == n) * (colormap[n][2])).astype('uint8')
    # color_result = Image.fromarray(np.uint8(result))
    return result

def save_img(preds, coarse_preds, edge_preds, scales, centers, input_size):
    classes = np.array(('Background',  # always index 0
                        'Hat', 'Hair', 'Glove', 'Sunglasses',
                        'UpperClothes', 'Dress', 'Coat', 'Socks',
                        'Pants', 'Jumpsuits', 'Scarf', 'Skirt',
                        'Face', 'Left-arm', 'Right-arm', 'Left-leg',
                        'Right-leg', 'Left-shoe', 'Right-shoe',))
    colormap = [(0, 0, 0),
                (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
                (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0),
                (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0),
                (128, 64, 0), (0, 192, 0), (128, 192, 0), ]
    colormap_binary = [(0, 0, 0), (255, 255, 255)]

    # print(x.shape)

    list_path = os.path.join(args.data_dir, args.dataset, 'id.txt')
    val_id = [i_id.strip() for i_id in open(list_path)]

    for i, im_name in enumerate(val_id):
        ori_img_path = os.path.join(args.data_dir, args.dataset, 'image', im_name + '.jpg')
        parsing_gt_path = os.path.join(args.data_dir, args.dataset, 'gt', im_name + '.png')
        edge_gt_path = os.path.join(args.data_dir, args.dataset, 'gt_edge', im_name + '.png')

        ori_img = cv2.imread(ori_img_path, cv2.IMREAD_COLOR)
        parsing_gt = cv2.imread(parsing_gt_path, cv2.IMREAD_GRAYSCALE)
        edge_gt = cv2.imread(edge_gt_path, cv2.IMREAD_GRAYSCALE)

        h, w = parsing_gt.shape
        s = scales[i]
        c = centers[i]

        if not os.path.exists('result/{}'.format(i)):
            os.makedirs('result/{}'.format(i))

        # ori_img = Image.fromarray(np.uint8(ori_img))
        # ori_img.save('result/{}/{}.png'.format(i, im_name))
        cv2.imwrite('result/{}/{}.png'.format(i, im_name), ori_img)

        pred_out = preds[i]
        pred = transform_parsing(pred_out, c, s, w, h, input_size)
        parsing_pred = np.asarray(pred, dtype=np.int32)
        color_parsing_pred = cam_mask(parsing_pred, colormap_binary, args.num_classes)
        # color_parsing_pred.save('result/{}/{}_parsing_pred.png'.format(i, im_name))
        cv2.imwrite('result/{}/{}_parsing_pred.png'.format(i, im_name), color_parsing_pred)

        coarse_pred_out = coarse_preds[i]
        coarse_pred = transform_parsing(coarse_pred_out, c, s, w, h, input_size)
        coarse_parsing_pred = np.asarray(coarse_pred, dtype=np.int32)
        color_coarse_parsing_pred = cam_mask(coarse_parsing_pred, colormap_binary, args.num_classes)
        # color_coarse_parsing_pred.save('result/{}/{}_coarse_parsing_pred.png'.format(i, im_name))
        cv2.imwrite('result/{}/{}_coarse_parsing_pred.png'.format(i, im_name), color_coarse_parsing_pred)

        edge_out = edge_preds[i]
        edge_pred = transform_parsing(edge_out, c, s, w, h, input_size)
        edge_pred = np.asarray(edge_pred, dtype=np.int32)
        color_edge_pred = cam_mask(edge_pred, colormap_binary, 2)
        # color_edge_pred.save('result/{}/{}_edge_pred.png'.format(i, im_name))
        cv2.imwrite('result/{}/{}_edge_pred.png'.format(i, im_name), color_edge_pred)

        parsing_gt = np.asarray(parsing_gt, dtype=np.int32)
        color_parsing_gt = cam_mask(parsing_gt, colormap_binary, args.num_classes)
        # color_parsing_gt.save('result/{}/{}_parsing_gt.png'.format(i, im_name))
        cv2.imwrite('result/{}/{}_parsing_gt.png'.format(i, im_name), color_parsing_gt)

        edge_gt = np.asarray(edge_gt, dtype=np.int32)
        edge_gt = cam_mask(edge_gt, colormap_binary, 2)
        # edge_gt.save('result/{}/{}_edge_gt.png'.format(i, im_name))
        cv2.imwrite('result/{}/{}_edge_gt.png'.format(i, im_name), edge_gt)

        # cv2.imwrite('result/{}.png'.format(im_name), color_pred)

    # x = x.squeeze(0)
    # x = F.softmax(torch.from_numpy(x), dim=0).argmax(0).cpu().numpy()
    # # x = F.softmax(x, dim=0).argmax(0).cpu().numpy().astype(np.float64)
    # # x = cv2.resize(x, (128, 128))
    # color_x = cam_mask(x, colormap, args.num_classes)
    #
    # cv2.imwrite('result/parsing_{}.png'.format(index), color_x)
    # # color_x.save()

def valid(model, valloader, input_size, num_samples, gpus):
    model.eval()

    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]), dtype=np.uint8)
    coarse_parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]), dtype=np.uint8)
    edge_preds = np.zeros((num_samples, input_size[0], input_size[1]), dtype=np.uint8)

    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in enumerate(valloader):
            image, meta = batch

            num_images = image.size(0)
            if index % 100 == 0:
                print('%d  processd' % (index * num_images))

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]

            outputs = model(image.cuda())
            if gpus > 1:
                for output in outputs:
                    parsing = output[0][-1]
                    nums = len(parsing)
                    parsing = interp(parsing).data.cpu().numpy()
                    parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                    parsing_preds[idx:idx + nums, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

                    idx += nums
            else:
                parsing = outputs[0][-1]
                # parsing_save = transform_parsing(parsing, centers, scales, 128, 128, input_size)
                # save_img(index, parsing_save)
                parsing = interp(parsing).data.cpu().numpy()
                parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

                coarse_parsing = outputs[0][0]
                coarse_parsing = interp(coarse_parsing).data.cpu().numpy()
                coarse_parsing = coarse_parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                coarse_parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(coarse_parsing, axis=3), dtype=np.uint8)

                edge = outputs[1][0]
                edge = interp(edge).data.cpu().numpy()
                edge = edge.transpose(0, 2, 3, 1)  # NCHW NHWC
                edge_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(edge, axis=3), dtype=np.uint8)

                idx += num_images

    parsing_preds = parsing_preds[:num_samples, :, :]
    coarse_parsing_preds = coarse_parsing_preds[:num_samples, :, :]
    edge_preds = edge_preds[:num_samples, :, :]
    return parsing_preds, coarse_parsing_preds, edge_preds, scales, centers

def main():
    """Create the model and start the evaluation process."""


    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    gpus = [int(i) for i in args.gpu.split(',')]

    h, w = map(int, args.input_size.split(','))
    
    input_size = (h, w)

    model = Res_Deeplab(num_classes=args.num_classes)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    lip_dataset = LIPDataSet(args.data_dir, 'val', crop_size=input_size, transform=transform)
    num_samples = len(lip_dataset)

    valloader = data.DataLoader(lip_dataset, batch_size=args.batch_size * len(gpus),
                                shuffle=False, pin_memory=True)

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

    parsing_preds,  coarse_parsing_preds, edge_preds, scales, centers = valid(model, valloader, input_size, num_samples, len(gpus))
    save_img(parsing_preds,  coarse_parsing_preds, edge_preds, scales, centers, input_size)

    # mIoU = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size)
    #
    # print(mIoU)

if __name__ == '__main__':
    main()
