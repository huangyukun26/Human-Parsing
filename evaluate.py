import argparse

from tqdm import tqdm
import numpy as np
import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
from networks.CE2P_PSD2_DA_new import Res_Deeplab
from dataset.datasets import LIPDataSet
import os
import torchvision.transforms as transforms
from utils.miou import compute_mean_ioU
from copy import deepcopy

# def get_arguments():
#     """Parse all the arguments provided from the CLI.
#
#     Returns:
#       A list of parsed arguments.
#     """
parser = argparse.ArgumentParser(description="CE2P Network")
parser.add_argument("--gpu", type=str, default='2', help="choose gpu device.")
parser.add_argument("--restore_from", type=str, default='./snapshots_256_256/demo/epoch_0.pth', help="Where restore model parameters from.")
parser.add_argument("--input_size", type=str, default='256,256', help="Comma-separated string with height and width of images.")
parser.add_argument("--batch_size", type=int, default=8, help="Number of images sent to the network in one step.")
parser.add_argument("--data_dir", type=str, default='./dataset/ATR', help="Path to the directory containing the PASCAL VOC dataset.")
parser.add_argument("--dataset", type=str, default='val', help="Path to the file listing the images in the dataset.")
parser.add_argument("--ignore_label", type=int, default=255, help="The index of the label to ignore during the training.")
parser.add_argument("--num_classes", type=int, default=20, help="Number of classes to predict (including background).")
args = parser.parse_args()

def valid(model, valloader, input_size, num_samples, gpus):
    model.eval()

    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]), dtype=np.uint8)

    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in enumerate(tqdm(valloader)):
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
                parsing = interp(parsing).data.cpu().numpy()
                parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

                idx += num_images

    parsing_preds = parsing_preds[:num_samples, :, :]
    return parsing_preds, scales, centers

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

    parsing_preds, scales, centers = valid(model, valloader, input_size, num_samples, len(gpus))

    mIoU = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size)
    print(mIoU)

if __name__ == '__main__':
    main()
