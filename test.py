import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from Unet import Generic_UNet, InitWeights_He

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
# General configures.
parser.add_argument('--name', default='Deepmedic', type=str, help='name of experiment')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# Test configures.
parser.add_argument('--saveresults', help='To save results in name', action='store_true')
parser.add_argument('--patch-size', default=[80,80,80], nargs='+', type=int, help='patch size')
# Network configures.
parser.add_argument('--downsampling', default=4, type=int, help='too see if I need deeper arch')
parser.add_argument('--features', default=30, type=int, help='feature map')
parser.add_argument('--deepsupervision', action='store_true', help='use deep supervision, just like nnunet')
# Dataset configures.
parser.add_argument('--liver0', default=0, type=float, help='choose the dataset')
parser.add_argument('--trainval', help='to test on the training data, just for debugging', action='store_true')
# CoLab parameters.
parser.add_argument('--taskcls', type=int, default=3, help='how many classes I expect for the task geneartor')
parser.set_defaults(augment=True)
args = parser.parse_args()

if __name__ == '__main__':

    if args.liver0 == 0:
        ## lits aux
        NumsInputChannel = 1
        NumsClass = args.taskcls
        from common_test import testlitstumor as test
        if args.trainval:
            DatafileValFold = './datafiles/liver0/traintumorset/'
        else:
            DatafileValFold = './datafiles/liver0/valtumor/'

    torch.cuda.set_device(args.gpu)
    # create model
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d
    conv_per_stage = 2
    base_num_features = args.features

    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
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

    model = Generic_UNet(NumsInputChannel, base_num_features, NumsClass,
                            len(net_num_pool_op_kernel_sizes),
                            conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                            dropout_op_kwargs,
                            net_nonlin, net_nonlin_kwargs, args.deepsupervision, False, lambda x: x, InitWeights_He(1e-2),
                            net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)
    model = model.cuda()
    # model.train()
    model.eval()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:' + str(args.gpu))
            # args.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    DSC, SENS, PREC = test(model, args.saveresults, args.name + '/results/', trainval=args.trainval, 
                        ImgsegmentSize=args.patch_size, deepsupervision=args.deepsupervision,
                        DatafileValFold=DatafileValFold, NumsClass=NumsClass)

    print('DSC ' + str(DSC))
    print('SENS ' + str(SENS))
    print('PREC ' + str(PREC))

    if len(DSC) > 1:
        print('DSCavg ' + str(np.mean(DSC[1:])))
        print('SENSavg ' + str(np.mean(SENS[1:])))
        print('PRECavg ' + str(np.mean(PREC[1:])))
