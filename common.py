import torch
import numpy as np
import torch.nn as nn
from utilities import SoftDiceLoss, accuracy, _concat, SoftPRECLoss, SoftSENSLoss, SoftSPECLoss, hessian_vector_product_Unet
from collections import OrderedDict
from Unet import Generic_UNet, InitWeights_He
from tensorboard_logger import log_value
from utilities import resize_segmentation
import time
import torch.nn.functional as F
from common_test import testlitstumor

def train(sampling_results, sampling_results_val, model, samplemodel, criterion, 
    optimizer, optimizer_arch, alpha, epoch, logging, args, bratsflag = False):

    logging.info(' **************** epoch ' + str(epoch) + ' is starting ****************')

    # decompose the results package.
    inputnor = sampling_results[0]
    target = sampling_results[1]
    _ = sampling_results[2]
    listr = np.array(sampling_results[-1])

    '''Train for one epoch on the training set'''
    batch_time = AverageMeter()
    losses = AverageMeter()
    lossvales = AverageMeter()
    top1 = AverageMeter()
    top1val = AverageMeter()

    lossvaldsces = AverageMeter()
    lossvalbcees = AverageMeter()
    lossvalpbcees = AverageMeter()
    lossvalnbcees = AverageMeter()
    lossvalprces = AverageMeter()
    lossvalsenes = AverageMeter()
    lossvalspees = AverageMeter()
    lossvalpzes = AverageMeter()
    lossvalnzes = AverageMeter()

    GeneratedTasks = AverageMeter()

    # switch to train mode
    model.train()
    samplemodel.train()
    end = time.time()

    for iteration in range(int(args.numIteration)):

        # process with training samples.
        targetpick = torch.tensor(target[iteration * args.batch_size: (iteration + 1) * args.batch_size, :, :, :])
        target_var = targetpick.long().cuda()
        target_var = torch.autograd.Variable(target_var)
        inputnorpick = torch.tensor(inputnor[iteration * args.batch_size: (iteration + 1) * args.batch_size, :, :, :])
        inputnor_var = inputnorpick.float().cuda()
        inputnor_var = torch.autograd.Variable(inputnor_var)

        targetpick_val = targetpick.clone()
        target_varval = target_var.clone()
        inputnorpick_val = inputnorpick.clone()
        inputnor_varval = inputnor_var.clone()
        
        if args.manuallabel:
            ## change the label in targetpickval
            target_varval[target_varval == 1] = 0
            target_varval[target_varval == 2] = 1

        if iteration % args.taskupdate == 0:

            # compute output
            '''
            I want to update the model one batch at a time.
            '''
            for kbatch in range(inputnor_var.shape[0]):
                inputnor_var_batch = inputnor_var[kbatch:kbatch + 1, :, :, :, :]
                target_var_batch = target_var[kbatch:kbatch + 1, :, :, :]
                inputnor_varval_batch = inputnor_varval[kbatch:kbatch + 1, :, :, :, :]
                target_varval_batch = target_varval[kbatch:kbatch + 1, :, :, :]

                output = model(inputnor_var_batch)
                taskGenerated = samplemodel(inputnor_var_batch)

                '''
                There is always a background class, I would to exclude that class when caclcualte the dsc loss.
                I should calculate the current prediction, to find a class that is very large, which is BG.
                '''
                # I should decide the background class based on taskGenerated
                Taskist = list(range(args.taskcls))
                Tasknumlist = np.zeros(len(Taskist))
                ## I wanna remove the largest class.
                Largeclsnum = torch.argmax(taskGenerated, dim = 1)
                for kcls in Taskist:
                    Tasknumlist[kcls] = torch.sum(Largeclsnum==kcls)
                
                GeneratedTasks.update(Tasknumlist, 1)

                avgres = GeneratedTasks.avg
                BGclsindex = np.argmax(avgres)
                ## I would assume the FG task is class 1, therefore I should add 1 to the class list
                BGclsindex += BGclsindex != 0

                if kbatch == 0:
                    logging.info('average BG class is ' + str(BGclsindex))

                '''the moving average ends here'''
                
                ## this is the pseudo label
                taskGenerateds = torch.softmax(taskGenerated, 1)
                
                '''try to restrict the region to update the model'''
                if args.distdetach:
                    threshold_sub = args.threshold_sub
                    # fetch the distmap
                    targetdistpick = torch.tensor(_[iteration * args.batch_size: (iteration + 1) * args.batch_size, :, :, :])
                    targetdistpick = targetdistpick[kbatch:kbatch + 1, :, :, :]
                    targetdist_var = targetdistpick.unsqueeze(1)

                    '''targetdist_var: [Batch, 1, H, W, D], distmap calculated with lesion label'''
                    distmap = - targetdist_var + threshold_sub
                    distmap = torch.exp(distmap / args.threshold_dev)
                    
                    distmap[distmap > 1] = 1
                    ## only consider the surranding 20 pixels.
                    # distmap[distmap < 1] = 0 ## maybe it is unneccessary

                    distmap = distmap.float()
                    distmap = distmap.cuda()
                    if args.deepsupervision:
                        loss_mask = []
                        for kds in range(args.downsampling):
                            if kds == 0:
                                loss_mask.append(distmap.detach())
                            else:
                                loss_mask_downsample = - F.max_pool3d(-distmap, stride=2**kds, kernel_size=2**kds).detach()
                                loss_mask.append(loss_mask_downsample)
                    else:
                        loss_mask = distmap.detach()
                else:
                    loss_mask = None
                # loss_mask = None
                ''''it ends here'''   

                if args.deepsupervision:
                    taskGeneratedall = []
                    for kds in range(args.downsampling):
                        if kds == 0:
                            taskGeneratedall.append(taskGenerateds)
                        else:
                            taskGeneratedall.append(F.avg_pool3d(taskGenerateds, stride=2**kds, kernel_size=2**kds))

                    ## to see when it degrade.
                    if iteration % args.print_freq == 0:
                        for kcls in range(args.taskcls):
                            portion0 = taskGeneratedall[0].data.cpu().numpy()
                            portion0 = np.argmax(portion0, axis=1)
                            portion0 = np.sum(portion0 == kcls) / np.size(portion0)
                            logging.info('Task 1 portion' + str(kcls) + ' ' + str(portion0))
                else:
                    taskGeneratedall = taskGenerateds

                # it might go too far, if I directly use deepsupervision here, but it is more straightforward for me.
                '''Now I have the output of time t.'''

                # print(len(Augweightpick))
                # just try use the normal training batch
                fast_weights = OrderedDict(model.named_parameters())

                losssample = calculate_loss(args, target_var_batch, output, taskGeneratedall, losstype = 0, loss_masks = loss_mask, BGcls = BGclsindex)

                gradients = torch.autograd.grad(losssample, fast_weights.values())
                # gradients = torch.autograd.grad(losssample, fast_weights.values(), retain_graph=True) 
                # if I have larger gpu, I can use retain_graph=True                                                                                                                                 

                fast_weights = OrderedDict(
                    (name, param - alpha * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )
                del gradients

                # logging.info('meta model forward ' + str(time.time() - point1) + ' seconds')  # 300s ? for the first time
                # point2 = time.time()

                theta = _concat(list(fast_weights.items())).data

                # create model, for the meta-learning process
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
                unrolled_model = Generic_UNet(args.NumsInputChannel, base_num_features, args.NumsClass + args.taskcls - 1,
                                    len(net_num_pool_op_kernel_sizes),
                                    conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, args.deepsupervision, False, lambda x: x, InitWeights_He(1e-2),
                                    net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

                unrolled_model = unrolled_model.cuda()
                model_dict = model.state_dict()
                params, offset = {}, 0
                for k, v in model.named_parameters():
                    v_length = np.prod(v.size())
                    params[k] = theta[offset: offset + v_length].view(v.size())
                    offset += v_length
                assert offset == len(theta)
                model_dict.update(params)
                unrolled_model.load_state_dict(model_dict)
                outputval = unrolled_model(inputnor_varval_batch)
                
                lossval, lossvalbce, lossvaldsc, lossvalpbce, lossvalnbce, lossvalpz, lossvalnz, lossvalprc, lossvalsen, lossvalspe = calcualte_loss_val(args, target_varval_batch, outputval, criterionbce = nn.BCELoss().cuda())
                # outputval 10,4,21,21,21/ target_val_var 10,21,21,21

                if (args.vanilla == False) and (args.manuallabel == False):
                    
                    skipgradflag = False

                    optimizer_arch.zero_grad()
                    lossval.backward()
                    ## the aux part would get zero gradients...So I should be careful here.
                    vector = []
                    for v in unrolled_model.parameters():
                        if v.grad is None:
                            vector.append(torch.zeros(v.shape).cuda())
                        else:
                            vector.append(v.grad.data)
                            '''caution: it would bring issues, could not be trained.'''
                            e1 = 0
                            if torch.max(torch.abs(v.grad.data)) <= e1 and v.shape[-1] != 1:
                                ## if it is the output layer, it would be fine, because I do not use them.
                                logging.info('skipping optimizing the meta-learner....')
                                skipgradflag = True
                                break

                    ## if vector is all zeros (use pbce) or vector is too small(training with mask)
                    ## ..then I should not calculate implicit_grads. otherwise we would expect a lot of nan.
                    if skipgradflag == False:
            
                        implicit_grads = hessian_vector_product_Unet(model, samplemodel, vector, inputnor_var_batch, target_var_batch, taskGeneratedall, criterion, args, loss_mask, BGclsindex)

                        for v, g in zip(samplemodel.parameters(), implicit_grads):
                            if v.grad is None:
                                v.grad = - alpha * g.data # it is the outer learning rate (for segmentor.)
                            else:
                                v.grad.data.copy_(- alpha * g.data)
                else:

                    del taskGenerated
                    del taskGeneratedall

             
                ############################################################################################################################################

                if (args.vanilla == False) and (args.manuallabel == False):
                    '''here I want to calcualte the entropy'''

                    optimizer_arch.step()

                # logging.info('meta model backward ' + str(time.time() - point2) + ' seconds')  # 300s ? for the first epoch
                # point3 = time.time()

        ######################### normal training #########################
        # I calcualte the output again, recreate the graph, to save memory.
        output = model(inputnor_var)
        # If I have larger gpu, I do not need this..for a second time.
        taskGenerated = samplemodel(inputnor_var)
        taskGenerateds = torch.softmax(taskGenerated, 1)
        
        '''try to restrict the region to update the model'''
        if args.distdetach:
            threshold_sub = args.threshold_sub
            # fetch the distmap
            targetdistpick = torch.tensor(_[iteration * args.batch_size: (iteration + 1) * args.batch_size, :, :, :])
            targetdist_var = targetdistpick.unsqueeze(1)

            '''targetdist_var: [Batch, 1, H, W, D], distmap calculated with lesion label'''
            distmap = - targetdist_var + threshold_sub
            distmap = torch.exp(distmap / args.threshold_dev)
            
            distmap[distmap > 1] = 1
            ## only consider the surranding 20 pixels.
            # distmap[distmap < 1] = 0 ## maybe it is unneccessary
            distmap = distmap.float()
            distmap = distmap.cuda()
            if args.deepsupervision:
                loss_mask = []
                for kds in range(args.downsampling):
                    if kds == 0:
                        loss_mask.append(distmap.detach())
                    else:
                        loss_mask_downsample = - F.max_pool3d(-distmap, stride=2**kds, kernel_size=2**kds).detach()
                        loss_mask.append(loss_mask_downsample)
            else:
                loss_mask = distmap.detach()
        else:
            loss_mask = None
        ''''it ends here'''           

        if args.deepsupervision:
            taskGenerateddet = []
            for kds in range(args.downsampling):
                if kds == 0:
                    taskGenerateddet.append(taskGenerateds.detach())
                else:
                    taskGenerateddet.append(F.avg_pool3d(taskGenerateds, stride=2**kds, kernel_size=2**kds).detach())
        else:
            taskGenerateddet = taskGenerateds.detach()

        '''
        It is a little cumbsome, if it uses deepsupervision, I calculate the loss like this.
        I repeat it for 2+1+2 times just in this script and Utilities, maybe it can be done in a better way.
        '''
        losssample = calculate_loss(args, target_var, output, taskGenerateddet, detach=True, logging=logging, loss_masks = loss_mask, BGcls = BGclsindex)
        if args.vanilla and iteration % args.print_freq == 0:
            outorigin = []
            for kds in range(args.downsampling):
                outtemp = output[kds]
                outorigin.append(outtemp[:, 0:args.NumsClass, :, : , :])
            logging.info('total loss:')
            losssample_origin = calculate_loss_origin(args, target_var, outorigin)
            logging.info(losssample)
            logging.info(losssample_origin)
            logging.info('CE loss:')
            losssample_ce = calculate_loss(args, target_var, output, taskGenerateddet, losstype = 1, BGcls = BGclsindex)
            losssample_origin_ce = calculate_loss_origin(args, target_var, outorigin, losstype = 1)
            logging.info(losssample_ce)
            logging.info(losssample_origin_ce)

        optimizer.zero_grad()
        losssample.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        optimizer.step()

        # logging.info('model backward ' + str(time.time() - point3) + ' seconds')  # 300s ? for the first time

        # measure accuracy and record loss
        if args.manuallabel:
            ## change the label in targetpick and ..
            targetpick[targetpick==1] = 0
            targetpick[targetpick==2] = 1
        if args.deepsupervision:
            outputdata = output[0].data
            outputvaldata = outputval[0].data
        else:
            outputdata = output.data
            outputvaldata = outputval.data
        for clsrm in range(args.NumsClass, args.NumsClass+args.taskcls-1):
            # this would make the acc not mean anything.
            outputdata[:, clsrm, :, :, :] = -100
            outputvaldata[:, clsrm, :, :, :] = -100
        prec1 = accuracy(outputdata, targetpick.long().cuda(), topk=(1,))[0]
        losses.update(losssample.data.item(), inputnorpick.size()[0])
        top1.update(prec1.item(), inputnorpick.size()[0])
        if iteration % args.taskupdate == 0 :
            prec1val = accuracy(outputvaldata, target_varval_batch, topk=(1,))[0]
            lossvales.update(lossval.data.item(), inputnorpick_val.size()[0])
            top1val.update(prec1val.item(), inputnorpick_val.size()[0])
            lossvaldsces.update(lossvaldsc.data.item(), inputnorpick_val.size()[0])
            lossvalbcees.update(lossvalbce.data.item(), inputnorpick_val.size()[0])
            lossvalpbcees.update(lossvalpbce.data.item(), inputnorpick_val.size()[0])
            lossvalnbcees.update(lossvalnbce.data.item(), inputnorpick_val.size()[0])
            lossvalpzes.update(lossvalpz.data.item(), inputnorpick_val.size()[0])
            lossvalnzes.update(lossvalnz.data.item(), inputnorpick_val.size()[0])
            lossvalprces.update(lossvalprc.data.item(), inputnorpick_val.size()[0])
            lossvalsenes.update(lossvalsen.data.item(), inputnorpick_val.size()[0])
            lossvalspees.update(lossvalspe.data.item(), inputnorpick_val.size()[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % args.print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Lossval {lossval.val:.4f} ({lossval.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, iteration, args.numIteration, batch_time=batch_time,
                loss=losses, lossval=lossvales, top1=top1))

    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_lossval', lossvales.avg, epoch)
        log_value('train_acc', top1.avg, epoch)
        log_value('trainval_acc', top1val.avg, epoch)
        log_value('train_lossvaldsc', lossvaldsces.avg, epoch)
        log_value('train_lossvalbce', lossvalbcees.avg, epoch)
        log_value('train_lossvalpbce', lossvalpbcees.avg, epoch)
        log_value('train_lossvalnbce', lossvalnbcees.avg, epoch)
        log_value('train_lossvalpz', lossvalpzes.avg, epoch)
        log_value('train_lossvalnz', lossvalnzes.avg, epoch)
        log_value('train_lossvalprc', lossvalprces.avg, epoch)
        log_value('train_lossvalsen', lossvalsenes.avg, epoch)
        log_value('train_lossvalspe', lossvalspees.avg, epoch)
    

def calculate_loss(args, targets, output, taskGenerated, detach = False, criterion = nn.CrossEntropyLoss().cuda(), losstype = 0, logging = None, loss_masks = None, BGcls = 0):
    '''
    This is the important function to calcualte the loss, both for meta-update and network update
    '''
    ## the smooth term here should be larger.
    e1 = 1e-32

    losssample = 0
    atasklist = list(range(args.NumsClass+args.taskcls-1))
    for krcls in list(range(args.NumsClass)):
        if krcls > 0:
            atasklist.remove(krcls)
            # it would be like [0, 2]
    
    FGlist = list(range(args.NumsClass+args.taskcls-1))
    FGlist.remove(BGcls)

    if args.deepsupervision:
        targetpicks = targets.data.cpu().numpy()
        weights = np.array([1 / (2 ** i) for i in range(args.downsampling)])
        mask = np.array([True] + [True if i < args.downsampling - 1 else False for i in range(1, args.downsampling)])
        weights[~mask] = 0
        weights = weights / weights.sum()
        for kds in range(args.downsampling):
            targetpickx = targetpicks[:, np.newaxis]
            s = np.ones(3) * 0.5 ** kds
            axes = list(range(2, len(targetpickx.shape)))
            new_shape = np.array(targetpickx.shape).astype(float)
            for i, a in enumerate(axes):
                new_shape[a] *= s[i]
            # in case it is something like 160 * 160 * 80
            if args.patch_size[1] != args.patch_size[2]:
                if kds > 0:
                    new_shape[4] = new_shape[4] * 2
            new_shape = np.round(new_shape).astype(int)
            out_targetpickx = np.zeros(new_shape, dtype=targetpickx.dtype)
            for b in range(targetpickx.shape[0]):
                for c in range(targetpickx.shape[1]):
                    out_targetpickx[b, c] = resize_segmentation(targetpickx[b, c], new_shape[2:], order=0, cval=0)
            # if would be very slow if I used tensor from the begining.
            target_varsx = torch.tensor(out_targetpickx[:, 0, :, :, :])

            target_varsx = target_varsx.long().cuda()
            target_varsx = torch.autograd.Variable(target_varsx)

            outputas = output[kds]
            taskGeneratedas = taskGenerated[kds]
            # taskGeneratedas has the shape of [N, C, H, W, D]
            # here I need to subtitude the GT into taskGenerated.
            y_aux_given_x_train = taskGeneratedas

            # Here I need to stack y_aux_given_x_train with GT of FG cls.
            if args.manuallabel:
                y_onehot = torch.zeros([outputas.shape[0], 3, outputas.shape[2], outputas.shape[3], outputas.shape[4]])
            else:
                y_onehot = torch.zeros([outputas.shape[0], args.NumsClass, outputas.shape[2], outputas.shape[3], outputas.shape[4]])
            y_onehot = y_onehot.cuda()

            target_label = target_varsx.view((target_varsx.shape[0], 1, *target_varsx.shape[1:]))
            y_onehot.scatter_(1, target_label, 1)
            y_aux_given_x_train = torch.cat((y_aux_given_x_train[:, 0:1, :, :, :], y_onehot[:, 1:args.NumsClass, :, :, :], y_aux_given_x_train[:, 1:, :, :, :]), 1)

            '''
            Here I want to combine the context label with the original labels.
            '''
            for kcls in list(range(y_aux_given_x_train.shape[1])):
                if args.vanilla:
                    # it would make generated tasks all zeros.
                    for kclsc in list(range(y_aux_given_x_train.shape[1])):
                        ctemp = y_aux_given_x_train[:, kclsc, :, :, :]
                        ctemp[target_varsx==kcls] = float(kcls == kclsc)
                else:
                    if kcls < args.NumsClass and kcls != 0:
                        ## Here I want to make sure the fg cls is not affected.
                        for kclsc in list(range(y_aux_given_x_train.shape[1])):
                            ctemp = y_aux_given_x_train[:, kclsc, :, :, :]
                            ctemp[target_varsx==kcls] = float(kcls == kclsc)
            # print(torch.sum(y_aux_given_x_train[:, 0:2, :, :, :] - y_onehot))
            # print(torch.sum(y_aux_given_x_train[:, 2, :, :, :]))

            if args.manuallabel:
                y_aux_given_x_train = y_onehot[:, [0, 2, 1], :, :, :]
            if loss_masks is not None:
                loss_mask = loss_masks[kds]
                ## here I want to make the loss_mask always contain the FG label.
                ## particularly for the downsampled GT.
                FGlabel = y_onehot[:, 1:args.NumsClass, :, :, :]
                loss_mask[(FGlabel.sum(axis=1, keepdims=True) == 1) & (loss_mask < 1)] = 1
            else:
                loss_mask = torch.ones(y_onehot.shape[0], 1, y_onehot.shape[2], y_onehot.shape[3], y_onehot.shape[4])
                loss_mask = loss_mask.cuda()
            '''
            I should notice that, loss_mask indicates the ones could be FG
            the pixels outside loss_mask are taken as BG, always.
            '''

            if losstype == 0:

                # this code might reduce the dsc we had before, because it is mean of different classes (exclude bg cls).
                if detach:
                    '''Maybe I do not want a large BG class'''
                    # losssample += weights[kds] * SoftDiceLoss(outputas, y_aux_given_x_train, list(range(args.NumsClass+args.taskcls-1)), loss_mask = loss_mask)
                    BGcls = 0
                    y_cls0 = torch.zeros(y_aux_given_x_train.shape)
                    y_cls0[:, BGcls, :, :, :] = 1
                    y_cls0 = y_cls0.cuda()
                    y_comb = y_aux_given_x_train * loss_mask + y_cls0 * (1 - loss_mask)
                    if args.vanilla:
                        losssample += weights[kds] * SoftDiceLoss(outputas, y_comb, list(range(args.NumsClass)))
                    else:
                        losssample += weights[kds] * SoftDiceLoss(outputas, y_comb, list(range(args.NumsClass+args.taskcls-1)))
                else:
                    '''What loss should I choose for inner iteration?
                    For the inner training process, I do not need to add DSC loss. CE loss would provide most contribution
                    '''
                    # losssample += weights[kds] * SoftDiceLoss(outputas, y_aux_given_x_train, list(range(args.NumsClass+args.taskcls-1)), loss_mask = loss_mask)
                    BGcls = 0
                    y_cls0 = torch.zeros(y_aux_given_x_train.shape)
                    y_cls0[:, BGcls, :, :, :] = 1
                    y_cls0 = y_cls0.cuda()
                    y_comb = y_aux_given_x_train * loss_mask + y_cls0 * (1 - loss_mask)
                    losssample += weights[kds] * SoftDiceLoss(outputas, y_comb, list(range(args.NumsClass+args.taskcls-1)))

            ## I calculate the loss myself here, because the default xentr loss requires hard vector 
            outputas = outputas.transpose(1, 2)
            outputas = outputas.transpose(2, 3)
            outputas = outputas.transpose(3, 4).contiguous()
            outputas = outputas.view(-1, outputas.shape[4])
            y_aux_given_x_train = y_aux_given_x_train.transpose(1, 2)
            y_aux_given_x_train = y_aux_given_x_train.transpose(2, 3)
            y_aux_given_x_train = y_aux_given_x_train.transpose(3, 4).contiguous()
            y_aux_given_x_train = y_aux_given_x_train.view(-1, y_aux_given_x_train.shape[4])
            
            p_y_given_x_train = torch.softmax(outputas, 1)
            log_p_y_given_x_train = (p_y_given_x_train + e1).log()
            # I should do bce for the FG cls

            ## the shape should be like [2, 1, 80, 80, 80]
            loss_mask = loss_mask.transpose(1, 2)
            loss_mask = loss_mask.transpose(2, 3)
            loss_mask = loss_mask.transpose(3, 4).contiguous()
            loss_mask = loss_mask.view(-1, loss_mask.shape[4])

            '''I just assume cls0 is the BG class, I am not sure if it is correct'''
            ## here it is N * class
            y_cls0 = torch.zeros(p_y_given_x_train.shape)
            y_cls0[:, BGcls] = 1
            y_cls0 = y_cls0.cuda()
            y_comb = y_aux_given_x_train * loss_mask + y_cls0 * (1 - loss_mask)

            lossCE = - (1. / p_y_given_x_train.shape[0]) * log_p_y_given_x_train * y_comb
            
            # lossCE = lossCE.sum()
            lossCEmain = lossCE[:, 1:args.NumsClass]
            lossauxgenerated = lossCE[:, atasklist]
            lossaux = lossCEmain.sum() + lossauxgenerated.sum()

            losssample += weights[kds] * lossaux
    else:
        outputas = output
        loss_mask = loss_masks
        taskGeneratedas = taskGenerated
        # here I need to subtitude the GT into taskGenerated.
        # I do not fullly understand what would happen with that..maybe leave is
        y_aux_given_x_train = taskGeneratedas
        y_onehot = torch.zeros([outputas.shape[0], args.NumsClass, outputas.shape[2], outputas.shape[3], outputas.shape[4]])
        y_onehot = y_onehot.cuda()
        target_label = targets.view((targets.shape[0], 1, *targets.shape[1:]))
        y_onehot.scatter_(1, target_label, 1)
        y_aux_given_x_train = torch.cat((taskGeneratedas[:, 0:1, :, :, :], y_onehot[:, 1:args.NumsClass, :, :, :], taskGeneratedas[:, 1:, :, :, :]), 1)
        for kcls in list(range(y_aux_given_x_train.shape[1])):
            if kcls < args.NumsClass and kcls != 0:
                ## there is a need to remake the label.
                for kclsc in list(range(y_aux_given_x_train.shape[1])):
                    ctemp = y_aux_given_x_train[:, kclsc, :, :, :]
                    ctemp[targets==kcls] = float(kcls == kclsc)
        
        if losstype == 0:
            
            # if detach and logging is not None:
            #     logging.info('BG class is ' + str(BGclsindex))

            # this code might reduce the dsc we had before, because it is mean of different classes (exclude bg cls).
            if detach:
                '''Maybe I do not want a large BG class'''
                BGcls = 0
                y_cls0 = torch.zeros(y_aux_given_x_train.shape)
                y_cls0[:, BGcls, :, :, :] = 1
                y_cls0 = y_cls0.cuda()
                y_comb = y_aux_given_x_train * loss_mask + y_cls0 * (1 - loss_mask)
                losssample += SoftDiceLoss(outputas, y_comb, list(range(args.NumsClass)))
            else:
                '''What loss should I choose for inner iteration?'''
                BGcls = 0
                y_cls0 = torch.zeros(y_aux_given_x_train.shape)
                y_cls0[:, BGcls, :, :, :] = 1
                y_cls0 = y_cls0.cuda()
                y_comb = y_aux_given_x_train * loss_mask + y_cls0 * (1 - loss_mask)
                losssample += SoftDiceLoss(outputas, y_comb, list(range(args.NumsClass+args.taskcls-1)))
        
        ## I calculate the loss myself here, because the default xentr loss requires hard vector 
        outputas = outputas.transpose(1, 2)
        outputas = outputas.transpose(2, 3)
        outputas = outputas.transpose(3, 4).contiguous()
        outputas = outputas.view(-1, outputas.shape[4])
        y_aux_given_x_train = y_aux_given_x_train.transpose(1, 2)
        y_aux_given_x_train = y_aux_given_x_train.transpose(2, 3)
        y_aux_given_x_train = y_aux_given_x_train.transpose(3, 4).contiguous()
        y_aux_given_x_train = y_aux_given_x_train.view(-1, y_aux_given_x_train.shape[4])

        p_y_given_x_train = torch.softmax(outputas, 1)
        log_p_y_given_x_train = (p_y_given_x_train + e1).log()

        ## the shape should be like [2, 1, 80, 80, 80]
        loss_mask = loss_mask.transpose(1, 2)
        loss_mask = loss_mask.transpose(2, 3)
        loss_mask = loss_mask.transpose(3, 4).contiguous()
        loss_mask = loss_mask.view(-1, loss_mask.shape[4])

        y_cls0 = torch.zeros(p_y_given_x_train.shape)
        y_cls0[:, BGcls] = 1
        y_cls0 = y_cls0.cuda()
        y_comb = y_aux_given_x_train * loss_mask + y_cls0 * (1 - loss_mask)

        lossCE = - (1. / p_y_given_x_train.shape[0]) * log_p_y_given_x_train * y_comb

        # lossaux = lossaux.sum()
        lossCEmain = lossCE[:, 1:args.NumsClass]
        lossauxgenerated = lossCE[:, atasklist]
        lossaux = lossCEmain.sum() + lossauxgenerated.sum()
        losssample += lossaux

    return losssample

def calculate_loss_origin(args, target_var, output, criterion = nn.CrossEntropyLoss().cuda(), losstype = 0):
    '''
    This is just an update function to make sure the calculated loss is identity to the original in some cases
    '''
    if args.deepsupervision:
        losssample = 0
        targetpicks = target_var.data.cpu().numpy()
        weights = np.array([1 / (2 ** i) for i in range(args.downsampling)])
        mask = np.array([True] + [True if i < args.downsampling - 1 else False for i in range(1, args.downsampling)])
        weights[~mask] = 0
        weights = weights / weights.sum()
        for kds in range(args.downsampling):
            targetpickx = targetpicks[:, np.newaxis]
            s = np.ones(3) * 0.5 ** kds
            axes = list(range(2, len(targetpickx.shape)))
            new_shape = np.array(targetpickx.shape).astype(float)
            for i, a in enumerate(axes):
                new_shape[a] *= s[i]
            # in case it is something like 160 * 160 * 80
            if args.patch_size[1] != args.patch_size[2]:
                if kds > 0:
                    new_shape[4] = new_shape[4] * 2
            new_shape = np.round(new_shape).astype(int)
            out_targetpickx = np.zeros(new_shape, dtype=targetpickx.dtype)
            for b in range(targetpickx.shape[0]):
                for c in range(targetpickx.shape[1]):
                    out_targetpickx[b, c] = resize_segmentation(targetpickx[b, c], new_shape[2:], order=0, cval=0)
            # if would be very slow if I used tensor from the begining.
            target_vars = torch.tensor(np.squeeze(out_targetpickx))

            if len(target_vars.size()) == 3:
                target_vars = target_vars.unsqueeze(0)

            target_vars = target_vars.long().cuda()
            target_vars = torch.autograd.Variable(target_vars)
            if losstype == 0:
                losssample += weights[kds] * (criterion(output[kds], target_vars) + 
                    SoftDiceLoss(output[kds], target_vars, list(range(args.NumsClass))))
            else:
                losssample += weights[kds] * criterion(output[kds], target_vars)
    else:
        if losstype == 0:
            losssample = SoftDiceLoss(output, target_var, list(range(args.NumsClass))) + criterion(output, target_var)
        else:
            losssample = criterion(output, target_var)
    
    return losssample

def calcualte_loss_val(args, target_varval, outputval, criterionbce):
    '''
    This is the function to calculate the meta-validation loss.
    I calculated a lot of validation values here, in order to assess the performance in a comprehensive way
    '''

    # larger smooth here, to make things stable.
    # if I set is as 1e-32, the loss would becomme nan. Maybe it is connected with the meta-gradient process.
    e1 = 1e-6

    lossvaldsc = 0
    lossvalprc = 0
    lossvalsen = 0
    lossvalspe = 0
    lossvalbce = 0
    lossvalpbce = 0
    lossvalnbce = 0
    lossvalpz = 0
    lossvalnz = 0

    lossval = 0
    dsclosslist = list(range(args.NumsClass))
    celossclist = list(range(1, args.NumsClass))
    if args.deepsupervision:
        outputas_val = outputval[0]
        p_y_given_x_train = torch.softmax(outputas_val, 1)
        y_onehot = torch.zeros([outputas_val.shape[0], args.NumsClass, outputas_val.shape[2], outputas_val.shape[3], outputas_val.shape[4]])
        y_onehot = y_onehot.cuda()
        target_label = target_varval.view((target_varval.shape[0], 1, *target_varval.shape[1:]))
        y_onehot.scatter_(1, target_label, 1)

        p = p_y_given_x_train[:, celossclist, :, :, :]
        y = y_onehot[:, celossclist, :, :, :]
        bcemap = -(p + e1).log() * y - (1 - p + e1).log() * (1-y)
        lossval += bcemap.sum() / (p.shape[0] * p.shape[2] * p.shape[3] * p.shape[4]) 
        
        # just for record, previously it is only for debug, but I leave it here, it should not take much time

        bcemap = -(p + e1).log() * y - (1 - p + e1).log() * (1-y)
        lossvalbce += bcemap.sum() / (p.shape[0] * p.shape[2] * p.shape[3] * p.shape[4]) 
        pbcemap = -(p + e1).log() * y
        lossvalpbce += pbcemap.sum() / (p.shape[0] * p.shape[2] * p.shape[3] * p.shape[4])
        nbcemap = - (1 - p + e1).log() * (1-y)
        lossvalnbce += nbcemap.sum() / (p.shape[0] * p.shape[2] * p.shape[3] * p.shape[4])

        atasklist = list(range(args.NumsClass+args.taskcls-1))
        for krcls in list(range(args.NumsClass)):
            if krcls > 0:
                atasklist.remove(krcls)
        logitBG = outputas_val[:, atasklist, :, :, :] * y_onehot[:, 0:1, :, :, :]
        logitBG, _ = torch.max(logitBG, dim = 1, keepdim = True)
        lossvalpz += logitBG.sum() / (logitBG.shape[0] * logitBG.shape[2] * logitBG.shape[3] * logitBG.shape[4])

        # logitBGn = outputas_val[:, atasklist, :, :, :] * y_onehot[:, 1:args.NumsClass, :, :, :]
        logitBGn = outputas_val[:, atasklist, :, :, :] * y_onehot[:, args.NumsClass-1:args.NumsClass, :, :, :]
        logitBGn, _ = torch.max(logitBGn, dim = 1, keepdim = True)
        lossvalnz += logitBGn.sum() / (logitBGn.shape[0] * logitBGn.shape[2] * logitBGn.shape[3] * logitBGn.shape[4])

        lossvaldsc += SoftDiceLoss(outputas_val, target_varval, list(range(args.NumsClass)))
        lossvalprc += SoftPRECLoss(outputas_val, target_varval, list(range(args.NumsClass)))
        lossvalsen += SoftSENSLoss(outputas_val, target_varval, list(range(args.NumsClass)))
        lossvalspe += SoftSPECLoss(outputas_val, target_varval, list(range(args.NumsClass)))
    else:
        outputas_val = outputval
        p_y_given_x_train = torch.softmax(outputas_val, 1)
        y_onehot = torch.zeros([outputas_val.shape[0], args.NumsClass, outputas_val.shape[2], outputas_val.shape[3], outputas_val.shape[4]])
        y_onehot = y_onehot.cuda()
        target_label = target_varval.view((target_varval.shape[0], 1, *target_varval.shape[1:]))
        y_onehot.scatter_(1, target_label, 1)
        p = p_y_given_x_train[:, celossclist, :, :, :]
        y = y_onehot[:, celossclist, :, :, :]
        bcemap = -(p + e1).log() * y - (1 - p + e1).log() * (1-y)
        lossval += bcemap.sum() / (p.shape[0] * p.shape[2] * p.shape[3] * p.shape[4]) 
        
        bcemap = -(p + e1).log() * y - (1 - p + e1).log() * (1-y)
        lossvalbce += bcemap.sum() / (p.shape[0] * p.shape[2] * p.shape[3] * p.shape[4]) 
        pbcemap = -(p + e1).log() * y
        lossvalpbce += pbcemap.sum() / (p.shape[0] * p.shape[2] * p.shape[3] * p.shape[4])
        nbcemap = - (1 - p + e1).log() * (1-y)
        lossvalnbce += nbcemap.sum() / (p.shape[0] * p.shape[2] * p.shape[3] * p.shape[4])

        atasklist = list(range(args.NumsClass+args.taskcls-1))
        for krcls in list(range(args.NumsClass)):
            if krcls > 0:
                atasklist.remove(krcls)
        logitBG = outputas_val[:, atasklist, :, :, :] * y_onehot[:, 0:1, :, :, :]
        logitBG, _ = torch.max(logitBG, dim = 1, keepdim = True)
        lossvalpz += logitBG.sum() / (logitBG.shape[0] * logitBG.shape[2] * logitBG.shape[3] * logitBG.shape[4])

        logitBGn = outputas_val[:, atasklist, :, :, :] * y_onehot[:, 1:args.NumsClass, :, :, :]
        logitBGn, _ = torch.max(logitBGn, dim = 1, keepdim = True)
        lossvalnz += logitBGn.sum() / (logitBGn.shape[0] * logitBGn.shape[2] * logitBGn.shape[3] * logitBGn.shape[4])

        lossvaldsc += SoftDiceLoss(outputas_val, target_varval, list(range(args.NumsClass)))
        lossvalprc += SoftPRECLoss(outputas_val, target_varval, list(range(args.NumsClass)))
        lossvalsen += SoftSENSLoss(outputas_val, target_varval, list(range(args.NumsClass)))
        lossvalspe += SoftSPECLoss(outputas_val, target_varval, list(range(args.NumsClass)))

    return lossval, lossvalbce, lossvaldsc, lossvalpbce, lossvalnbce, lossvalpz, lossvalnz, lossvalprc, lossvalsen, lossvalspe

def adjust_learning_rate(optimizer, epoch, args):
    # and more to converge
    lr = args.lr * (1 - epoch / args.epochs)**0.9

    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_arch(optimizer_arch, epoch, args):
    """it is used by meta-task."""
    # and more to converge
    archlr = args.arch_learning_rate * (1 - epoch / args.epochs)**0.9
    
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate_arch', archlr, epoch)
    for param_group in optimizer_arch.param_groups:
        param_group['lr'] = archlr

def validatelitstumor(DatafileValFold, model, criterion, logging, epoch, Savename, args, NumsClass = 2):
    model.eval()

    DSC, SENS, PREC = testlitstumor(model, True, Savename + '/results/',
                            ImgsegmentSize=args.patch_size,
                            deepsupervision=args.deepsupervision, DatafileValFold=DatafileValFold, NumsClass = NumsClass)

    logging.info('DSC ' + str(DSC))
    logging.info('SENS ' + str(SENS))
    logging.info('PREC ' + str(PREC))
    # log to TensorBoard
    if args.tensorboard:
        log_value('DSCtumor', DSC[0], epoch)
        log_value('SENStumor', SENS[0], epoch)
        log_value('PRECtumor', PREC[0], epoch)
    return DSC.mean()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count