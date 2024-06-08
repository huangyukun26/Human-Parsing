import argparse

import numpy as np
import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
from networks.CE2P_PSD2_DA_new import Res_Deeplab
from dataset.datasets import LIPDataSet
import torchvision.transforms as transforms
import timeit
from tensorboardX import SummaryWriter
from utils.criterion import CriterionAll
from utils.encoding import DataParallelModel, DataParallelCriterion 
from utils.logger import setup_logger
from tqdm import tqdm

parser = argparse.ArgumentParser(description="CE2P Network")
parser.add_argument("--num-classes", type=int, default=20, help="Number of classes to predict (including background).")
parser.add_argument("--snapshot_dir", type=str, default='./snapshots_256_256/demo', help="Where to save snapshots of the model.")
parser.add_argument("--gpu", type=str, default='2', help="choose gpu device.")
parser.add_argument("--input_size", type=str, default='256,256',
                    help="Comma-separated string with height and width of images.")
parser.add_argument("--start_epoch", type=int, default=0, help="choose the number of recurrence.")
parser.add_argument("--epochs", type=int, default=1, help="choose the number of recurrence.")
parser.add_argument("--batch_size", type=int, default=8, help="Number of images sent to the network in one step.")
parser.add_argument("--data_dir", type=str, default='./dataset/ATR', help="Path to the directory containing the dataset.")
parser.add_argument("--dataset", type=str, default='train', choices=['train', 'val', 'trainval', 'val'],
                    help="Path to the file listing the images in the dataset.")
parser.add_argument("--ignore_label", type=int, default=255, help="The index of the label to ignore during the training.")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Base learning rate for training with polynomial decay.")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
parser.add_argument("--start_iters", type=int, default=0, help="Number of classes to predict (including background).")
parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate.")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="Regularisation parameter for L2-loss.")
parser.add_argument("--random_mirror", action="store_true", help="Whether to randomly mirror the inputs during the training.")
parser.add_argument("--random_scale", action="store_true", help="Whether to randomly scale the inputs during the training.")
parser.add_argument("--random_seed", type=int, default=1234, help="Random seed to have reproducible results.")
parser.add_argument("--restore_from", type=str, default='pretrained/resnet101-imagenet.pth', help="Where restore model parameters from.")
parser.add_argument("--save_num_images", type=int, default=2, help="How many images to save.")

args = parser.parse_args()

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, total_iters):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, total_iters, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def main():
    """Create the model and start the training."""

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    logger = setup_logger(args.snapshot_dir, if_train=True)
    logger.info(args.snapshot_dir)
    # logger.info("use gpu:", args.gpu)
    # logger.info("input_size:", args.input_size)
    # logger.info("batch_size:", args.batch_size)


    writer_dir = os.path.join(args.snapshot_dir.split('/')[0], 'events', args.snapshot_dir.split('/')[1]+'_events')
    writer = SummaryWriter(writer_dir)
    gpus = [int(i) for i in args.gpu.split(',')]
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]

    cudnn.enabled = True
    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    deeplab = Res_Deeplab(num_classes=args.num_classes)

    saved_state_dict = torch.load(args.restore_from)
    new_params = deeplab.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        # print(i_parts)
        if not i_parts[0] == 'fc':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

    deeplab.load_state_dict(new_params)
   
    model = DataParallelModel(deeplab)
    model.cuda()

    criterion = CriterionAll()
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    trainloader = data.DataLoader(LIPDataSet(args.data_dir, args.dataset, crop_size=input_size, transform=transform),
                                  batch_size=args.batch_size * len(gpus), shuffle=True, num_workers=8, pin_memory=True)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    optimizer.zero_grad()
    logger.info('--------------------start training--------------------')
    total_iters = args.epochs * len(trainloader)
    flag_loss = np.inf
    for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
        start_epoch = timeit.default_timer()
        epoch_loss = 0.0
        model.train()
        for i_iter, batch in enumerate(tqdm(trainloader)):
            i_iter += len(trainloader) * epoch
            lr = adjust_learning_rate(optimizer, i_iter, total_iters)

            images, labels, edges, _ = batch
            images = images.cuda()
            labels = labels.long().cuda(non_blocking=True)
            edges = edges.long().cuda(non_blocking=True)

            preds = model(images)

            loss = criterion(preds, [labels, edges])
            epoch_loss = epoch_loss + loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_iter % 100 == 0:
                writer.add_scalar('learning_rate', lr, i_iter)
                writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)
                logger.info('step = [{}/{}], loss={:0.4f}, lR={:0.7f}'.format(i_iter, total_iters, loss.data.cpu().numpy(), lr))
        epoch_loss = epoch_loss / len(trainloader)
        end_epoch = timeit.default_timer()
        logger.info("Epoch {} done. Time per epoch: {:.3f}[s] ".format(epoch+1, end_epoch-start_epoch))
        if epoch_loss < flag_loss:
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'epoch_' + str(epoch+1) + '.pth'))
            flag_loss = epoch_loss
            logger.info("Obtain the optimal model at the {}th epoch".format(epoch+1))
 

if __name__ == '__main__':
    main()
