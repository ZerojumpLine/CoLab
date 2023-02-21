import torch
import shutil
import numpy as np
import torch.nn as nn
from skimage.transform import resize
from torch import Tensor
from typing import Iterable, Tuple, Set
from scipy.ndimage import distance_transform_edt as distance

def get_tn_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tn = (1 - net_output) * (1 - y_onehot)
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tn = tn ** 2
        fp = fp ** 2
        fn = fn ** 2

    tn = sum_tensor(tn, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tn, fp, fn

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn

def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def SoftDiceLoss(x, y, clschosen, loss_mask = None, smooth = 1e-5, do_bg = False, batch_dice = False):
    '''
    Batch_dice means that we want to calculate the dsc of all batch
    It would make more sense for small patchsize, aka DeepMedic based training.
    '''
    shp_x = x.shape
    apply_nonlin = softmax_helper
    square = False

    if batch_dice:
        axes = [0] + list(range(2, len(shp_x)))
    else:
        axes = list(range(2, len(shp_x)))

    if apply_nonlin is not None:
        x = apply_nonlin(x)

    tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, square)

    dc = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

    if not do_bg:
        if batch_dice:
            clschosen.remove(0)
            dc_process = dc[clschosen]
        else:
            dc_process = []
            for ksel in clschosen:
                if ksel != 0 :
                    dc_process.append(dc[:, int(ksel)])
            dc_process = torch.cat(dc_process)
        dc_process = dc_process.mean()
    else:
        if batch_dice:
            dc_process = dc[clschosen]
        else:
            dc_process = []
            for ksel in clschosen:
                dc_process.append(dc[:, int(ksel)])
            dc_process = torch.cat(dc_process)
        dc_process = dc_process.mean()

    return -dc_process

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]

    return y[labels]  # [N,D]

def convert_seg_image_to_one_hot_encoding_batched(image, classes=None):
    '''
    same as convert_seg_image_to_one_hot_encoding, but expects image to be (b, x, y, z) or (b, x, y)
    '''
    if classes is None:
        classes = np.unique(image)
    output_shape = [image.shape[0]] + [len(classes)] + list(image.shape[1:])
    out_image = np.zeros(output_shape, dtype=image.dtype)
    for b in range(image.shape[0]):
        for i, c in enumerate(classes):
            out_image[b, i][image[b] == c] = 1
    return out_image

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def ComputMetric(ACTUAL, PREDICTED):
    ACTUAL = ACTUAL.flatten()
    PREDICTED = PREDICTED.flatten()
    idxp = ACTUAL == True
    idxn = ACTUAL == False

    tp = np.sum(ACTUAL[idxp] == PREDICTED[idxp])
    tn = np.sum(ACTUAL[idxn] == PREDICTED[idxn])
    fp = np.sum(idxn) - tn
    fn = np.sum(idxp) - tp
    FPR = fp / (fp + tn)
    if tp == 0 :
        dice = 0
        Precision = 0
        Sensitivity = 0
    else:
        dice = 2 * tp / (2 * tp + fp + fn)
        Precision = tp / (tp + fp)
        Sensitivity = tp / (tp + fn)
    return dice, Sensitivity, Precision

    f, axarr = plt.subplots(6, 10, figsize=(15, 15), gridspec_kw={"wspace": 0, "hspace": 0})
    for idx, ax in enumerate(f.axes):
        ax.imshow(images[idx])
        ax.axis("off")
        if titles: ax.set_title(titles[idx])
    plt.show()

def _concat(xs):
  return torch.cat([x[1].view(-1) for x in xs])

def _concatmodel(xs):
  return torch.cat([x.view(-1) for x in xs])

def hessian_vector_product_Unet(model, samplemodel, vector, inputnor_var, target_var, taskGenerated, criterion, args, loss_mask, BGclsindex, r=1e-2):

    from common import calculate_loss

    R = r / _concatmodel(vector).norm()

    for p, v in zip(model.parameters(), vector):
        p.data.add_(R, v)
    outputprimeaux = model(inputnor_var)

    losssampleprime = calculate_loss(args, target_var, outputprimeaux, taskGenerated, loss_masks = loss_mask, BGcls = BGclsindex)
    grads_p = torch.autograd.grad(losssampleprime, samplemodel.parameters(), retain_graph=True)

    for p, v in zip(model.parameters(), vector):
        p.data.sub_(2 * R, v)
    
    outputprimeaux = model(inputnor_var)
    losssampleprime = calculate_loss(args, target_var, outputprimeaux, taskGenerated, loss_masks = loss_mask, BGcls = BGclsindex)
    
    grads_n = torch.autograd.grad(losssampleprime, samplemodel.parameters())
    # it shares a lot with the large batch (700), always retain gradph

    for p, v in zip(model.parameters(), vector):
        p.data.add_(R, v)

    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

def clip_grad_value_(parameters, clip_value):
    """Clips gradient of an iterable of parameters at specified value.
    Gradients are modified in-place.
    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float or int): maximum allowed value of the gradients.
            The gradients are clipped in the range
            :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in parameters:
        # logging.info(p.data.max())
        p.data.clamp_(min=-clip_value, max=clip_value)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    ## output, N, C, H, W, D to N, C
    ## target N, H, W, D to N,

    target = target.view(-1)
    output = output.permute(0, 2, 3, 4, 1).contiguous()
    output = output.view(-1, output.shape[4])

    maxk = max(topk)
    batch_size = target.size(0)


    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in')

def resize_segmentation(segmentation, new_shape, order=3, cval=0):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped

def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)
    from batchgenerators_local.augmentations.utils import rotate_coords_3d, rotate_coords_2d
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        # it should consider both directions, I think.
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, -rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, -rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, -rot_z)), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, -rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)

def removesmallprob(policyprob_aug, samplingthreshold, samplenum):

    aug_pool = np.arange(len(policyprob_aug))
    policyprob_chosenindex = policyprob_aug > samplingthreshold

    selind_aug = np.random.choice(aug_pool[policyprob_chosenindex], samplenum)

    return selind_aug

def save_checkpoint(state, is_best, dataset, Savename, filename='checkpoint.pth.tar', record=0):
    """Saves checkpoint to disk"""
    directory = "./output/%s/%s/"%(dataset, Savename)
    if record > 0 and state['epoch'] % record == 0:
        filename = directory + str(state['epoch']) + filename
    else:
        filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './output/%s/%s/'%(dataset, Savename) + 'model_best.pth.tar')

def save_checkpoint_meta(state,  taskG_state, is_best, dataset, Savename, filename='checkpoint.pth.tar', record=0):
    """Saves checkpoint to disk"""
    directory = "./output/%s/%s/"%(dataset, Savename)
    if record > 0 and state['epoch'] % record == 0:
        filename = directory + str(state['epoch']) + filename
        taskgenerator_filename = directory + str(state['epoch']) + 'TaskGenerator.pth.tar'
    else:
        filename = directory + filename 
        taskgenerator_filename = directory + 'TaskGenerator.pth.tar'
    torch.save(state, filename)
    torch.save(taskG_state, taskgenerator_filename)
    if is_best:
        shutil.copyfile(filename, './output/%s/%s/'%(dataset, Savename) + 'model_best.pth.tar')

def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None) -> np.ndarray:
    # assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = distance(negmask, sampling=resolution) * negmask \
                - (distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res

def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h, d = seg.shape  # type: Tuple[int, int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h, d)
    assert one_hot(res)

    return res

def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
    return coords

def SoftPRECLoss(x, y, cls, loss_mask = None, smooth = 1e-5):
    shp_x = x.shape
    batch_dice = False
    apply_nonlin = softmax_helper
    square = False
    do_bg = False

    if batch_dice:
        axes = [0] + list(range(2, len(shp_x)))
    else:
        axes = list(range(2, len(shp_x)))

    if apply_nonlin is not None:
        x = apply_nonlin(x)

    tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, square)

    dc = (tp + smooth) / (tp + fp + smooth)

    if not do_bg:
        if batch_dice:
            dc = dc[1:]
        else:
            dc = dc[:, 1:]

    dc_process = []
    for ksel in cls:
        if ksel != 0 :
            dc_process.append(dc[:, int(ksel)-1])

    dc_process = torch.cat(dc_process)
    dc_process = dc_process.mean()

    return -dc_process

def SoftSPECLoss(x, y, cls, loss_mask = None, smooth = 1e-5):
    shp_x = x.shape
    batch_dice = False
    apply_nonlin = softmax_helper
    square = False
    do_bg = False

    if batch_dice:
        axes = [0] + list(range(2, len(shp_x)))
    else:
        axes = list(range(2, len(shp_x)))

    if apply_nonlin is not None:
        x = apply_nonlin(x)

    tn, fp, fn = get_tn_fp_fn(x, y, axes, loss_mask, square)

    dc = (tn + smooth) / (tn + fp + smooth)

    if not do_bg:
        if batch_dice:
            dc = dc[1:]
        else:
            dc = dc[:, 1:]

    dc_process = []
    for ksel in cls:
        if ksel != 0 :
            dc_process.append(dc[:, int(ksel)-1])

    dc_process = torch.cat(dc_process)
    dc_process = dc_process.mean()

    return -dc_process

def SoftSENSLoss(x, y, cls, loss_mask = None, smooth = 1e-5):
    shp_x = x.shape
    batch_dice = False
    apply_nonlin = softmax_helper
    square = False
    do_bg = False

    if batch_dice:
        axes = [0] + list(range(2, len(shp_x)))
    else:
        axes = list(range(2, len(shp_x)))

    if apply_nonlin is not None:
        x = apply_nonlin(x)

    tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, square)

    dc = (tp + smooth) / (tp + fn + smooth)

    if not do_bg:
        if batch_dice:
            dc = dc[1:]
        else:
            dc = dc[:, 1:]

    dc_process = []
    for ksel in cls:
        if ksel != 0 :
            dc_process.append(dc[:, int(ksel)-1])

    dc_process = torch.cat(dc_process)
    dc_process = dc_process.mean()

    return -dc_process

def SoftFPLoss(x, y, cls, loss_mask = None, smooth = 1e-5):
    shp_x = x.shape
    batch_dice = False
    apply_nonlin = softmax_helper
    square = False
    do_bg = False

    if batch_dice:
        axes = [0] + list(range(2, len(shp_x)))
    else:
        axes = list(range(2, len(shp_x)))

    if apply_nonlin is not None:
        x = apply_nonlin(x)

    tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, square)

    dc = - fp

    if not do_bg:
        if batch_dice:
            dc = dc[1:]
        else:
            dc = dc[:, 1:]

    dc_process = []
    for ksel in cls:
        if ksel != 0 :
            dc_process.append(dc[:, int(ksel)-1])

    dc_process = torch.cat(dc_process)
    dc_process = dc_process.mean()

    return -dc_process