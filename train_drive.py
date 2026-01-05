import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader

from networks.model import SwinTransformer
from utils import ramps, losses
from dataloaders.dataset import MyDataSet
from utils.util import compute_sdf

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='DTC/1consis_weight', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=8000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='maximum epoch number to train')
parser.add_argument('--D_lr', type=float,  default=1e-4,
                    help='maximum discriminator learning rate to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=16, help='random seed')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency_weight', type=float,  default=0.1,
                    help='balance factor to control supervised loss and consistency loss')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--beta', type=float,  default=0.3,
                    help='balance factor to control regional and sdm loss')
parser.add_argument('--gamma', type=float,  default=0.5,
                    help='balance factor to control supervised and consistency loss')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="kl", help='consistency_type')
parser.add_argument('--with_cons', type=str,
                    default="without_cons", help='with or without consistency')
parser.add_argument('--consistency', type=float,
                    default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model_DTC/" + args.exp + \
    "_{}swin_drive_truewindow{}/".format(
        args.labelnum, args.beta)


batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code1'):
        shutil.rmtree(snapshot_path + '/code1')
    shutil.copytree('.', snapshot_path + '/code1',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = SwinTransformer()
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    # training_set_path_img = r'/home/smh/Downloads/data/DRIVE1/train_np/img/'
    # training_set_path_lab = r'/home/smh/Downloads/data/DRIVE1/train_np/lab/'
    training_set_path_img = r'/opt/data/private/dataset/DRIVE/train/img256'
    training_set_path_lab = r'/opt/data/private/dataset/DRIVE/train/lab256'

    db_train = MyDataSet(training_set_path_img, training_set_path_lab)

    labelnum = args.labelnum    # default 16
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, 80))

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train,batch_size=args.batch_size,drop_last=True)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, (volume_batch,label_batch,HaveLabel) in enumerate(trainloader):
            # print("i_batch:", i_batch)
            # print("HaveLabel:", HaveLabel)
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            # volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            # volume_batch = torch.unsqueeze(volume_batch,dim=1)
            label_batch = torch.unsqueeze(label_batch,dim=1)

            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs_tanh, outputs = model(volume_batch)
            outputs_tanh = torch.unsqueeze(outputs_tanh, dim=1)
            outputs_soft = torch.sigmoid(outputs)

            # calculate the loss
            with torch.no_grad():
                gt_dis = compute_sdf(label_batch.cpu(
                ).numpy(), outputs.shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()
            if HaveLabel:
                loss_sdf = mse_loss(outputs_tanh, gt_dis)
                loss_seg = ce_loss(
                    outputs, label_batch.float())
                loss_seg_dice = losses.dice_loss(
                    outputs_soft, label_batch)
                dis_to_mask = torch.sigmoid(-1500 * outputs_tanh)

                consistency_loss = torch.mean((dis_to_mask - outputs_soft) ** 2)
                supervised_loss = loss_seg_dice + args.beta * loss_sdf
                consistency_weight = get_current_consistency_weight(iter_num // 150)

                loss = supervised_loss + consistency_weight * consistency_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                dis_to_mask = torch.sigmoid(-1500 * outputs_tanh)

                consistency_loss = torch.mean((dis_to_mask - outputs_soft) ** 2)
                consistency_weight = get_current_consistency_weight(iter_num // 150)

                loss = consistency_weight * consistency_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            iter_num = iter_num + 1

            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 500 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                # logging.info("save model_DTC to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
