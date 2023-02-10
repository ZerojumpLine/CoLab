import os
import nibabel as nib
import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from utilities import ComputMetric
from typing import Tuple, List

def testlitstumor(model, saveresults, name, trainval = False, ImgsegmentSize = [80, 160, 160], deepsupervision=False, DatafileValFold=None, tta=False, ttalist = [0], ttalistprob=[1], NumsClass = 2):
    batch_size = 1
    NumsInputChannel = 1
    if trainval == False:
        DatafileFold = DatafileValFold
        DatafileImgc1 = DatafileFold + 'Imgpre-eval.txt'
        DatafileLabel = DatafileFold + 'seg-eval.txt'
        DatafileMask = DatafileFold + 'mask-eval.txt'
    else:
        DatafileFold = DatafileValFold
        DatafileImgc1 = DatafileFold + 'Imgpre-train.txt'
        DatafileLabel = DatafileFold + 'seg-train.txt'
        DatafileMask = DatafileFold + 'mask-train.txt'

    Imgfilec1 = open(DatafileImgc1)
    Imgreadc1 = Imgfilec1.read().splitlines()
    Maskfile = open(DatafileMask)
    Maskread = Maskfile.read().splitlines()
    Labelfile = open(DatafileLabel)
    Labelread = Labelfile.read().splitlines()

    DSClist = []
    SENSlist = []
    PREClist = []
    PredSumlist = []

    for numr in range(len(Imgreadc1)):
    # for numr in range(50, 51):

        Imgnamec1 = Imgreadc1[numr]
        Imgloadc1 = nib.load(Imgnamec1)
        Imgc1 = Imgloadc1.get_fdata()
        Maskname = Maskread[numr]
        Maskload = nib.load(Maskname)
        roi_mask = Maskload.get_fdata()
        Labelname = Labelread[numr]
        Labelload = nib.load(Labelname)
        gtlabel = Labelload.get_fdata()

        Imgc1 = np.float32(Imgc1)

        knamelist = Imgnamec1.split("/")

        kname = knamelist[-1][0:-7]

        channels = Imgc1[None, ...]

        hp_results = tta_rolling(model, channels, batch_size, ImgsegmentSize, NumsInputChannel, NumsClass, tta, ttalist, ttalistprob, deepsupervision)
        
        predSegmentation = np.argmax(hp_results, axis=0)

        '''This is debuggin code for mask softmax'''
        # y_eye = np.eye(2)
        # y_onehot = y_eye[gtlabel.astype(int)]
        # y_onehot = np.transpose(y_onehot, (3,0,1,2))
        # y_onehot_repeat = np.repeat(y_onehot, 2, axis=0)
        # print(y_onehot_repeat[:, 0, 0, 0])
        # hp_results = np.exp(hp_results) / np.sum(np.exp(hp_results), axis=0)
        # hp_results = hp_results * y_onehot_repeat
        # hp_results[hp_results==0] = 100
        # predSegmentation = np.argmin(hp_results, axis=0)

        # hp_results = np.exp(hp_results) / np.sum(np.exp(hp_results), axis=0)
        # predSegmentation = hp_results[0, :, :, :]
        # print(np.sum(predSegmentation))

        ## use the mask to constratin the results
        PredSegmentationWithinRoi = predSegmentation * roi_mask
        # PredSegmentationWithinRoi = predSegmentation
        # sio.savemat('./result.mat', {'results': PredSegmentationWithinRoi})
        imgToSave = PredSegmentationWithinRoi

        if saveresults:
            npDtype = np.dtype(np.float32)
            proxy_origin = nib.load(Imgnamec1)
            hdr_origin = proxy_origin.header
            affine_origin = proxy_origin.affine
            proxy_origin.uncache()

            newLabelImg = nib.Nifti1Image(imgToSave, affine_origin)
            newLabelImg.set_data_dtype(npDtype)

            dimsImgToSave = len(imgToSave.shape)
            newZooms = list(hdr_origin.get_zooms()[:dimsImgToSave])
            if len(newZooms) < dimsImgToSave:  # Eg if original image was 3D, but I need to save a multi-channel image.
                newZooms = newZooms + [1.0] * (dimsImgToSave - len(newZooms))
            newLabelImg.header.set_zooms(newZooms)

            directory = "../output/litstumor/%s/" % (name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            savename = directory + 'pred_' + kname + '_Segm.nii.gz'
            nib.save(newLabelImg, savename)

        labelwt = gtlabel == 1
        predwt = imgToSave == 1

        DSCwt, SENSwt, PRECwt = ComputMetric(labelwt, predwt)
        # print(DSCwt)
        DSClist.append([DSCwt])
        SENSlist.append([SENSwt])
        PREClist.append([PRECwt])
        PredSumlist.append(np.sum(labelwt))
        print('case ' + str(numr) + ' done')

    sel = [n for n, i in enumerate(PredSumlist) if i > 0]

    DSClist = np.array(DSClist)
    DSCmean = DSClist[sel, :].mean(axis=0)
    SENSlist = np.array(SENSlist)
    SENSmean = SENSlist[sel, :].mean(axis=0)
    PREClist = np.array(PREClist)
    PRECmean = PREClist[sel, :].mean(axis=0)

    return DSCmean, SENSmean, PRECmean

def tta_rolling(model, channels, batch_size, ImgsegmentSize, NumsInputChannel, NumsClass, tta, ttalist, ttalistprob, deepsupervision):
    hp_results = 0

    for ttaindex, ttaindexprob in zip(ttalist, ttalistprob):
        

        channels_per_path = []
        channels_per_path.append(channels.copy())

        channels_augment = channels_per_path[0]

        offset = [0, 0, 0]

        pad_border_mode = 'constant'
        pad_kwargs = dict()
        pad_kwargs['constant_values'] = 0
        data, slicer = pad_nd_image(channels_augment, ImgsegmentSize, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape
        step_size = 0.5
        steps = _compute_steps_for_sliding_window(ImgsegmentSize, data_shape[1:], step_size)

        hp = np.zeros([NumsClass] + list(data.shape[1:]), dtype=np.float32)
        aggregated_nb_of_predictions = np.zeros([NumsClass] + list(data.shape[1:]), dtype=np.float32)
        xpixels = []
        ypixels = []
        zpixels = []

        for jx in steps[0]:
            for jy in steps[1]:
                for jz in steps[2]:
                    xpixels.append(jx)
                    ypixels.append(jy)
                    zpixels.append(jz)

        inputxnor = getallbatch(data, ImgsegmentSize, xpixels, ypixels, zpixels, offset)

        ## gaussian filter
        patch_size = [ImgsegmentSize[0], ImgsegmentSize[1], ImgsegmentSize[2]]
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * 1. / 8 for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        inputxnor = torch.tensor(np.array(inputxnor))
        inputxnor = inputxnor.float().cuda()
        for xlist in range(0, len(inputxnor), batch_size):
            batchxnor = inputxnor[xlist: xlist + batch_size, :, :, :, :]

            xstarts = xpixels[xlist: xlist + batch_size]
            ystarts = ypixels[xlist: xlist + batch_size]
            zstarts = zpixels[xlist: xlist + batch_size]

            with torch.no_grad():
                pred = model(batchxnor)
                ## in case it is multi-task model.
                if len(pred) == 2:
                    pred = pred[0]
                if deepsupervision:
                    output = pred[0]
                else:
                    output = pred
            output = output.data.cpu().numpy()

            kbatch = 0
            for xstart, ystart, zstart in zip(xstarts, ystarts, zstarts):
                # only crop the center parts.
                # maybe I should use gaussain? to do ..
                # hp[:, xstart + offset:xstart + offset + PredSizetest, ystart + offset:ystart + offset + PredSizetest,
                # zstart + offset:zstart + offset + PredSizetest] = output[kbatch, :, offset:offset + PredSizetest,
                #                                                   offset:offset + PredSizetest,
                #                                                   offset:offset + PredSizetest]
                hp[:, xstart:xstart + ImgsegmentSize[0], ystart:ystart + ImgsegmentSize[1],
                zstart:zstart + ImgsegmentSize[2]] += output[kbatch, :, :, :, :] * gaussian_importance_map
                aggregated_nb_of_predictions[:, xstart:xstart + ImgsegmentSize[0], ystart:ystart + ImgsegmentSize[1],
                zstart:zstart + ImgsegmentSize[2]] += gaussian_importance_map
                kbatch = kbatch + 1

        slicer = tuple(
            [slice(0, hp.shape[i]) for i in
             range(len(hp.shape) - (len(slicer) - 1))] + slicer[1:])
        hp = hp[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]
        hp = hp / aggregated_nb_of_predictions

        ## to see if the probability map needs revert spatial transformations 
        hp_revert = hp
        
        ## hp_revert with shape: 1, D, W, H
        ## Here I want to try to ensemble with prediction score.
        # hp_revert = np.exp(hp_revert) / np.sum(np.exp(hp_revert), axis=0)
        hp_results += hp_revert / sum(ttalistprob) * ttaindexprob
    
    return hp_results

def getallbatch(Imgenlarge, ImgsegmentSize, xpixels, ypixels, zpixels, offset):

    inputxnor = []

    # normal pathway
    for (selindex_x, selindex_y, selindex_z) in zip(xpixels, ypixels, zpixels):
        coord_center = np.zeros(3, dtype=int)
        coord_center[0] = selindex_x + ImgsegmentSize[0] // 2
        coord_center[1] = selindex_y + ImgsegmentSize[1] // 2
        coord_center[2] = selindex_z + ImgsegmentSize[2] // 2

        samplekernal_primary = 1
        channs_of_sample_per_path_normal = Imgenlarge[:,
                                    coord_center[0] - ImgsegmentSize[0] // 2: coord_center[0] + ImgsegmentSize[0] // 2,
                                    coord_center[1] - ImgsegmentSize[1] // 2: coord_center[1] + ImgsegmentSize[1] // 2,
                                    coord_center[2] - ImgsegmentSize[2] // 2: coord_center[2] + ImgsegmentSize[2] // 2]
        inputxnor.append(channs_of_sample_per_path_normal)

    return inputxnor

def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    """
    one padder to pad them all. Documentation? Well okay. A little bit

    :param image: nd image. can be anything
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
    len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
    the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
    Example:
    image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
    image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).

    :param mode: see np.pad for documentation
    :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
    to original shape
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
    divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation
    """
    if kwargs is None:
        kwargs = {'constant_values': 0}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        res = np.pad(image, pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer

def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]:
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 32 and step_size of 0.5, then we want to make 4 steps starting at coordinate 0, 27, 55, 78
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps