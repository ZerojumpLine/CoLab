import argparse
import os
import shutil
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
from multiprocessing.pool import ThreadPool
from Unet import Generic_UNet, InitWeights_He
from common import train, adjust_learning_rate, adjust_learning_rate_arch
from utilities import save_checkpoint_meta
from sampling import getbatchkitsatlas as getbatch
# used for logging to TensorBoard
from tensorboard_logger import configure
os.environ['KMP_WARNINGS'] = 'off'

parser = argparse.ArgumentParser(description='PyTorch CoLab Training')
# General configures.
parser.add_argument('--name', default='CoLab_Training', type=str, help='name of experiment')
parser.add_argument('--print-freq', '-p', default=20, type=int, help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, help='path to latest main network checkpoint (default: none)')
parser.add_argument('--resumetask', default='', type=str, help='path to task generator checkpoint (default: none)')
parser.add_argument('--startover', action='store_true', help='do not care about the training info in the ckpt')
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# Training configures.
parser.add_argument('--epochs', default=2000, type=int, help='number of total epochs to run')
parser.add_argument('--batch-size', default=2, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--numIteration', default=100, type=int, help='num of iteration per epoch')
parser.add_argument('--lr', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for task generator')
parser.add_argument('--maxsample', type=float, default=50, help='sample from cases, large number leads to longer time')
parser.add_argument('--evalevery', type=float, default=200, help='evaluation every epoches')
parser.add_argument('--FGportion', default=0.5, type=float, help='Portion of FG samples during the sampling process')
parser.add_argument('--record', default=100, type=int, help='save every ckpt of the training process')
parser.add_argument('--det', action='store_true', help='control seed to for control experiments')
# Network configures.
parser.add_argument('--downsampling', default=4, type=int, help='too see if I need deeper arch')
parser.add_argument('--features', default=30, type=int, help='the number of feature map')
parser.add_argument('--patch-size', default=[80,80,80], nargs='+', type=int, help='patch size')
parser.add_argument('--deepsupervision', action='store_true', help='use deep supervision to train the network')
# Dataset configures.
parser.add_argument('--liver0', default=0, type=float, help='choose the dataset')
parser.add_argument('--split', default=0, type=int, help='0 for vanilla training, 1 for training with manual GT labels')
# CoLab parameters.
parser.add_argument('--vanilla', action='store_true', help='do not use the generated ones.')
parser.add_argument('--manuallabel', action='store_true', help='I have and use liver masks in this case.')
parser.add_argument('--distdetach', action='store_true', help='only use the GT which is near the target')
parser.add_argument('--threshold_sub', default=0, type=float, help='reg distance')
parser.add_argument('--threshold_dev', default=20, type=float, help='to contorl the smoothness of the mask')
parser.add_argument('--taskcls', type=int, default=3, help='how many classes I expect for the task geneartor')
parser.add_argument('--taskupdate', default=5, type=int, help='I update the task network every this epoch, to save time.')


args = parser.parse_args()
best_prec1 = 0

if args.det:
    np.random.seed(79)
    torch.manual_seed(79)
    torch.cuda.manual_seed_all(79)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    ## note that I can only control the sampling case, but not the sampling patches (it is controled by a global seed).
else:
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False

def main():
    global best_prec1

    # some dataset specific configs.
    if args.liver0 == 0:
        dataset = 'litstumor'
        from common import validatelitstumor as validate
        if args.split == 0 : ## vanilla training
            DatafileTrainqueueFold = './datafiles/liver0/traintumorset/'
        if args.split == 1 : ## training with liver mask
            DatafileTrainqueueFold = './datafiles/liver0/trainset/'
        DatafileValFold = './datafiles/liver0/valtumor/'
        args.NumsInputChannel = 1
        args.NumsClass = 2

    Savename = args.name + 'taskcls' + str(args.taskcls)
    directory = "./output/%s/%s/"%(dataset, Savename)

    if not os.path.exists(directory):
        os.makedirs(directory)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(directory, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    if args.tensorboard: configure("./output/%s/%s"%(dataset, Savename))

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()
    start_epoch = 0

    # create model
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d
    conv_per_stage = 2
    base_num_features = args.features

    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': False}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': False}
    net_num_pool_op_kernel_sizes = []
    if args.patch_size[1] != args.patch_size[2]:
        net_num_pool_op_kernel_sizes.append([2, 2, 1])
        for kiter in range(0, args.downsampling - 1):  # (0,5)
            net_num_pool_op_kernel_sizes.append([2, 2, 2])
    else:
        for kiter in range(0, args.downsampling):  # (0,5)
            net_num_pool_op_kernel_sizes.append([2, 2, 2])
    net_conv_kernel_sizes = []
    for kiter in range(0, args.downsampling + 1):  # (0,6)
        net_conv_kernel_sizes.append([3, 3, 3])

    model = Generic_UNet(args.NumsInputChannel, base_num_features, args.NumsClass + args.taskcls - 1,
                        len(net_num_pool_op_kernel_sizes),
                        conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                        dropout_op_kwargs,
                        net_nonlin, net_nonlin_kwargs, args.deepsupervision, False, lambda x: x, InitWeights_He(1e-2),
                        net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)
    samplemodel = Generic_UNet(args.NumsInputChannel, base_num_features, args.taskcls,
                        len(net_num_pool_op_kernel_sizes),
                        conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                        dropout_op_kwargs,
                        net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                        net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    optimizer_arch = torch.optim.Adam(samplemodel.parameters(), lr=args.arch_learning_rate, betas=(0.9, 0.999), weight_decay=0)

    # get the number of model parameters
    logging.info('Number of model parameters: {} MB'.format(sum([p.data.nelement() for p in model.parameters()])/1e6))

    model = model.cuda()
    samplemodel = samplemodel.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:' + str(args.gpu))
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_prec1 = checkpoint['best_prec1']
            # when started, it needs initalization of prec1.
            prec1 = 0
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.resumetask:
        if os.path.isfile(args.resumetask):
            logging.info("=> loading task generator checkpoint '{}'".format(args.resumetask))
            checkpoint = torch.load(args.resumetask, map_location='cuda:' + str(args.gpu))
            # args.start_epoch = checkpoint['epoch']
            samplemodel.load_state_dict(checkpoint['state_dict'])
            if 'optimizer_arch' in checkpoint:
                optimizer_arch.load_state_dict(checkpoint['optimizer_arch'])
            logging.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resumetask, checkpoint['epoch']))
        else:
            logging.info("=> no task generator checkpoint found at '{}'".format(args.resumetask))

    # multiprocess
    mp_pool = None
    mp_pool = ThreadPool(processes=1)
    for epoch in range(start_epoch, args.epochs):
        alpha = adjust_learning_rate(optimizer, epoch, args)
        # Here I want to adjust learningrate of the generated task.
        adjust_learning_rate_arch(optimizer_arch, epoch, args)

        # do the sampling here.

        Augindex = np.zeros((10, args.batch_size * args.numIteration))
        sampling_results_val = None
        if mp_pool is None:
            # sequence processing
            # sample more validation cases in one iteration
            '''
            maybe I do not need inputnor_val
            '''
            sampling_results = getbatch(DatafileTrainqueueFold, args.batch_size, args.numIteration, Augindex,
                                                    args.maxsample, 0, logging, 0, args.patch_size, args.FGportion, args.manuallabel, None)
        elif epoch == start_epoch:  # Not previously submitted in case of first epoch
            # to get the sampling from the multiprocess. the sampling parameters might have mismatch
            sampling_results = getbatch(DatafileTrainqueueFold, args.batch_size, args.numIteration, Augindex,
                                                    args.maxsample, 0, logging, 0, args.patch_size, args.FGportion, args.manuallabel, None)
            sampling_job = mp_pool.apply_async(getbatch, (DatafileTrainqueueFold, args.batch_size, args.numIteration, Augindex,
                                                                args.maxsample, 0, logging, 0, args.patch_size, args.FGportion, args.manuallabel, None))
        elif epoch == args.epochs - 1: # last iteration
            # do not need to submit job
            sampling_results = sampling_job.get()
            mp_pool.close()
            mp_pool.join()
        else:
            # get old job and submit new job
            sampling_results = sampling_job.get()
            sampling_job = mp_pool.apply_async(getbatch, (DatafileTrainqueueFold, args.batch_size, args.numIteration, Augindex,
                                                                    args.maxsample, 0, logging, 0, args.patch_size, args.FGportion, args.manuallabel, None))
        # input shape N, H, W, D, Channl
        # target shape N, H, W, D, Class

        # train for one epoch
        train(sampling_results, sampling_results_val, model, samplemodel, criterion, 
            optimizer, optimizer_arch, alpha, epoch, logging, args)

        # evaluate on validation set every 5 epoches
        if args.evalevery > 0:
            if epoch % args.evalevery == 0 or epoch == args.epochs-1 :
                prec1 = validate(DatafileValFold, model, criterion, logging, epoch, Savename, args, NumsClass = args.NumsClass + args.taskcls - 1)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint_meta({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': best_prec1,
        }, {
            'epoch': epoch + 1,
            'state_dict': samplemodel.state_dict(),
            'optimizer_arch': optimizer_arch.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, dataset, Savename, record = args.record)
    logging.info('Best overall DSCuracy: %s ', best_prec1)

if __name__ == '__main__':
    main()