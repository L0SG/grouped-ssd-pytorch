from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
#from data import VOCroot, VOC_CLASSES as labelmap
from PIL import Image
# from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import FISHdetection, BaseTransform
from utils.augmentations import SSDAugmentation
import torch.utils.data as data
from ssd_multiphase_custom_group import build_ssd
import numpy as np
import h5py
from layers.box_utils import nms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sklearn.model_selection import train_test_split, KFold
import time

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_group_vanilla_BN_fusex1only_3negpos10000_CV3.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
#parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test1.txt'
    num_images = len(testset)
    for idx in range(num_images):
        # pull img & annotations
        img = testset.pull_image(idx)
        # only use portal phase annotation
        annotation = [[testset.pull_anno(idx)][0][2]]
        # base transform the image and permute to [phase, channel, h, w] and flatten to [1, channel, h, w]
        x = torch.from_numpy(transform(img)[0]).permute(0, 3, 1, 2).contiguous()
        x = x.view(-1, x.shape[2], x.shape[3])
        x = Variable(x, volatile=True).unsqueeze(0)

        with open(filename, mode='a') as f:
            #f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
            for box in annotation:
                f.write('label: '+' || '.join(str(b) for b in box)+'\n')
        if cuda:
            x = x.cuda()
        # tic
        t = time.time()
        y = net(x)      # forward pass
        # toc
        elapsed = time.time() - t
        print('Testing image {:d}/{:d}...inference time: {:f}'.format(idx + 1, num_images, elapsed))
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[2], img.shape[1],
                             img.shape[2], img.shape[1]])
        pred_num = 0

        coords_list = []
        # skip background class (0) with range start of 1
        for i in range(1, detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
#                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                coords_list.append(coords)
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: lesion ' + ' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1
        # calculate gt & pred bbox coords
        # (x, y) start & delta of ground truth
        xs_gt, ys_gt = annotation[0][0], annotation[0][1]
        xd_gt, yd_gt = annotation[0][2] - xs_gt, annotation[0][3] - ys_gt

        # visualization: draw gt & predicted bounding box and save to image
        # use portal phase (idx=2 at dim=0) and middle slice (idx=1 at dim=3)
        output_image = img[2, :, :, 1].copy()
        fig, ax = plt.subplots()
        ax.imshow(output_image, cmap='gray')
        plt.axis('off')

        # green gt box
        rect_gt = patches.Rectangle((xs_gt, ys_gt), xd_gt, yd_gt, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect_gt)

        # red pred box
        # (y, x) start & delta of prediction
        # do it only if the prediction exist
        if 'coords' in locals():
            for idx in range(len(coords_list)):
                xs_p, ys_p = int(coords_list[idx][0]), int(coords_list[idx][1])
                xd_p, yd_p = int(coords_list[idx][2]) - xs_p, int(coords_list[idx][3]) - ys_p
                rect_p = patches.Rectangle((xs_p, ys_p), xd_p, yd_p, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect_p)
        plt.savefig(os.path.join(args.save_folder, 'test_'+str(idx)+'.png'))
        plt.close()



if __name__ == '__main__':
    """"########## Data Loading & dimension matching ##########"""
    # load custom CT dataset
    datapath = '/home/vision/tkdrlf9202/Datasets/liver_lesion_aligned/lesion_dataset_4phase_aligned.h5'
    train_sets = [('liver_lesion')]
    cross_validation = 5
    cv_idx_for_test = 3

    def load_lesion_dataset(data_path):
        """
        loads custom liver dataset
        if preprocessed h5 data exists, load it
        if not, load and preprocess raw liver dataset
        :param data_path:
        :return: flattened CT and mask data
        """
        # check if the preprocessed dataset exists
        if os.path.isfile(data_path):
            # load the preprocessed dataset dump
            print('loading lesion dataset...')
            with h5py.File(data_path, 'r') as dataset_h5:
                group_ct = dataset_h5['ct']
                group_coordinate = dataset_h5['coordinate']
                ct = [i[:] for i in group_ct.values()]
                coordinate = [i[:] for i in group_coordinate.values()]
                dataset_h5.close()

        return ct, coordinate

    ct, coord = load_lesion_dataset(datapath)
    # ct: [subjects, sample, phase, channel, 512, 512]
    # coord: [subjects, sample, phase, channel, 5], [x_min, y_min, x_max, y_max, 0 (lesion class label)] format
    # make channels last & 0~255 uint8 image
    for idx in range(len(ct)):
        ct[idx] = np.transpose(ct[idx] * 255, [0, 1, 3, 4, 2]).astype(dtype=np.uint8)
        # use only coordinate from the middle slice, ditch the upper & lower ones
        coord[idx] = coord[idx][:, :, 1, :]

    # 5-fold CV
    kf = KFold(n_splits=cross_validation)
    kf.get_n_splits(ct, coord)

    # flatten the subject & sample dimension for each sets by stacking
    ct_train = []
    ct_valid = []
    coord_ssd_train = []
    coord_ssd_valid = []
    for train_index, valid_index in kf.split(ct):
        ct_train_part = [ct[x] for ind, x in enumerate(train_index)]
        ct_valid_part = [ct[x] for ind, x in enumerate(valid_index)]
        coord_train_part = [coord[x] for ind, x in enumerate(train_index)]
        coord_valid_part = [coord[x] for ind, x in enumerate(valid_index)]

        ct_train.append(np.vstack(ct_train_part))
        ct_valid.append(np.vstack(ct_valid_part))
        coord_ssd_train.append(np.vstack(coord_train_part).astype(np.float64))
        coord_ssd_valid.append(np.vstack(coord_valid_part).astype(np.float64))
    """
    # split train & valid set, subject-level (without shuffle)
    ct_train, ct_valid, coord_ssd_train, coord_ssd_valid = train_test_split(ct, coord, test_size=0.1, shuffle=False)

    # flatten the subject & sample dimension for each sets by stacking
    ct_train = np.vstack(ct_train)
    ct_valid = np.vstack(ct_valid)
    coord_ssd_train = np.vstack(coord_ssd_train).astype(np.float64)
    coord_ssd_valid = np.vstack(coord_ssd_valid).astype(np.float64)
    """
    """#########################################################"""


    # load net
    num_classes = 2 # 1 lesion + 1 background
    size = 300
    batch_norm = True

    net = build_ssd('test', size, num_classes, batch_norm=batch_norm) # initialize SSD
    # load weights trained with Dataparallel
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    state_dict = torch.load(args.trained_model)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    # eval mode
    net.eval()
    print('Finished loading model!')
    # load data
    #testset = VOCDetection(args.voc_root, [('2007', 'test')], None, AnnotationTransform())

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    # cv_idx_for_test must be equal to the checkpoint idx of the trained model (otherwise cheat)
    means = (34, 34, 34)
    trainset = FISHdetection(ct_train[cv_idx_for_test], coord_ssd_train[cv_idx_for_test], None, 'lesion_train')
    validset = FISHdetection(ct_valid[cv_idx_for_test], coord_ssd_valid[cv_idx_for_test], None, 'lesion_valid')

    # allset = FISHdetection(np.vstack(ct), np.vstack(coord).astype(np.float64), None, 'lesion_all')

    test_net(args.save_folder, net, args.cuda, validset,
             BaseTransform(size, means),
             thresh=args.visual_threshold)