import argparse
import torch
torch.multiprocessing.set_start_method("spawn", force=True)

import numpy as np
import torch.optim as optim
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import os
import os.path as osp

from dataset.datasets import LIPDataSet
import torchvision.transforms as transforms
import timeit
# from tensorboardX import SummaryWriter
from utils.utils import decode_parsing, inv_preprocess
from utils.criterion import CriterionAll
from utils.encoding import DataParallelModel, DataParallelCriterion
from utils.logger import setup_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


from networks.CE2P_PSD2_DA_new import Res_Deeplab

start = timeit.default_timer()

parser = argparse.ArgumentParser(description="CE2P Network")
parser.add_argument("--num-classes", type=int, default=20, help="Number of classes to predict (including background).")
parser.add_argument("--snapshot_dir", type=str, default='snapshots_384/ce2p_psd2_da_bs8_2',
                    help="Where to save snapshots of the model.")
parser.add_argument("--gpu", type=str, default='1', help="choose gpu device.")
parser.add_argument("--input-size", type=str, default='384,384',
                    help="Comma-separated string with height and width of images.")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument("--start-epoch", type=int, default=0, help="choose the number of recurrence.")
parser.add_argument("--epochs", type=int, default=150, help="choose the number of recurrence.")
parser.add_argument("--batch-size", type=int, default=4, help="Number of images sent to the network in one step.")
parser.add_argument("--data-dir", type=str, default='data/LIP', help="Path to the directory containing the dataset.")
parser.add_argument("--dataset", type=str, default='train', choices=['train', 'val', 'trainval', 'val'],
                    help="Path to the file listing the images in the dataset.")
parser.add_argument("--ignore-label", type=int, default=255,
                    help="The index of the label to ignore during the training.")
parser.add_argument("--learning-rate", type=float, default=1e-3,
                    help="Base learning rate for training with polynomial decay.")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
parser.add_argument("--start-iters", type=int, default=0, help="Number of classes to predict (including background).")
parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate.")
parser.add_argument("--weight-decay", type=float, default=5e-4, help="Regularisation parameter for L2-loss.")
parser.add_argument("--random-mirror", action="store_true",
                    help="Whether to randomly mirror the inputs during the training.")
parser.add_argument("--random-scale", action="store_true",
                    help="Whether to randomly scale the inputs during the training.")
parser.add_argument("--random-seed", type=int, default=1234, help="Random seed to have reproducible results.")
parser.add_argument("--restore-from", type=str, default='pretrained/resnet101-imagenet.pth',
                    help="Where restore model parameters from.")
parser.add_argument("--save-num-images", type=int, default=2, help="How many images to save.")

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
    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    logger = setup_logger(args.snapshot_dir, if_train=True)
    logger.info(args.snapshot_dir)

    # writer_dir = os.path.join(args.snapshot_dir.split('/')[0], 'events', args.snapshot_dir.split('/')[1] + '_events')
    # writer = SummaryWriter(writer_dir)
    # gpus = [int(i) for i in args.gpu.split(',')]
    # if not args.gpu == 'None':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]

    cudnn.enabled = True
    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    """加载数据"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    lip_dataset = LIPDataSet(args.data_dir, args.dataset, crop_size=input_size, transform=transform)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    trainloader = DataLoader(lip_dataset, sampler=train_sampler(lip_dataset, shuffle=True),
                             batch_size=args.batch_size, # * len(gpus),
                             shuffle=False, num_workers=2, pin_memory=True)

    """加载模型"""
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    deeplab = Res_Deeplab(num_classes=args.num_classes)

    if args.local_rank == 0:
        torch.distributed.barrier()

    saved_state_dict = torch.load(args.restore_from)
    new_params = deeplab.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        # print(i_parts)
        if not i_parts[0] == 'fc':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

    deeplab.load_state_dict(new_params)

    model=deeplab
    # model = DDP(deeplab, device_ids=[rank])
    # model = DataParallelModel(deeplab)
    model.to(args.device)

    criterion = CriterionAll()
    # criterion = DataParallelCriterion(criterion)
    criterion.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.local_rank !=-1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    if args.world_size > 1:
        train_epoch = 0
        trainloader.sampler.set_epoch(train_epoch)

    optimizer.zero_grad()
    logger.info('--------------------start training--------------------')
    total_iters = args.epochs * len(trainloader)
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        model.train()
        for i_iter, batch in enumerate(trainloader):
            if args.world_size > 1:
                train_epoch+=1
                trainloader.sampler.set_epoch(train_epoch)

            i_iter += len(trainloader) * epoch
            lr = adjust_learning_rate(optimizer, i_iter, total_iters)

            images, labels, edges, _ = batch
            images = images.to(args.device)
            labels = labels.long().to(args.device) #(non_blocking=True)
            edges = edges.long().to(args.device) #(non_blocking=True)

            preds = model(images)

            loss = criterion(preds, [labels, edges])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_iter % 100 == 0:
                # writer.add_scalar('learning_rate', lr, i_iter)
                # writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)

                # if i_iter % 500 == 0:
                #
                #     images_inv = inv_preprocess(images, args.save_num_images)
                #     labels_colors = decode_parsing(labels, args.save_num_images, args.num_classes, is_pred=False)
                #     edges_colors = decode_parsing(edges, args.save_num_images, 2, is_pred=False)
                #
                #     if isinstance(preds, list):
                #         preds = preds[0]
                #     preds_colors = decode_parsing(preds[0][-1], args.save_num_images, args.num_classes, is_pred=True)
                #     pred_edges = decode_parsing(preds[1][-1], args.save_num_images, 2, is_pred=True)
                #
                #     img = vutils.make_grid(images_inv, normalize=False, scale_each=True)
                #     lab = vutils.make_grid(labels_colors, normalize=False, scale_each=True)
                #     pred = vutils.make_grid(preds_colors, normalize=False, scale_each=True)
                #     edge = vutils.make_grid(edges_colors, normalize=False, scale_each=True)
                #     pred_edge = vutils.make_grid(pred_edges, normalize=False, scale_each=True)
                #
                #     writer.add_image('Images/', img, i_iter)
                #     writer.add_image('Labels/', lab, i_iter)
                #     writer.add_image('Preds/', pred, i_iter)
                #     writer.add_image('Edges/', edge, i_iter)
                #     writer.add_image('PredEdges/', pred_edge, i_iter)

                # print('Epoch{}: iter = {} of {} completed, loss = {}'.format(str(epoch), i_iter, total_iters, loss.data.cpu().numpy()))
                logger.info(
                    'step = [{}/{}], loss={:0.4f}, lR={:0.7f}'.format(i_iter, total_iters, loss.data.cpu().numpy(), lr))

        end_epoch = timeit.default_timer()
        logger.info("Epoch {} done. Time per epoch: {:.3f}[s] ".format(epoch, end_epoch - start))
        if args.local_rank == 0:
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'LIP_epoch_' + str(epoch) + '.pth'))

    end = timeit.default_timer()
    print(end - start, 'seconds')
    logger('Training used a total of', end - start, 'seconds')


if __name__ == '__main__':
    main()
