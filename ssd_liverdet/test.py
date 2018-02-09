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
from ssd_multiphase_custom import build_ssd
import numpy as np
import h5py
from layers.box_utils import nms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='/home/vision/tkdrlf9202/PycharmProjects/liver_segmentation/ssd_liverdet/weights/ssd300_allconv_25000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=False, type=bool,
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
        print('Testing image {:d}/{:d}....'.format(idx+1, num_images))
        # pull img & annotations
        img = testset.pull_image(idx)
        annotation = [testset.pull_anno(idx)]
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

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
#                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: lesion ' + ' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1

        # calculate gt & pred bbox coords
        # (x, y) start & delta of ground truth
        xs_gt, ys_gt = annotation[0][0], annotation[0][1]
        xd_gt, yd_gt = annotation[0][2] - xs_gt, annotation[0][3] - ys_gt
        # (y, x) start & delta of prediction
        # do it only if the prediction exist
        if 'coords' in locals():
            xs_p, ys_p = int(coords[0]), int(coords[1])
            xd_p, yd_p = int(coords[2]) - xs_p, int(coords[3]) - ys_p

        # visualization: draw gt & predicted bounding box and save to image
        output_image = img.copy()
        fig, ax = plt.subplots(1)
        ax.imshow(output_image)
        # green gt box
        rect_gt = patches.Rectangle((xs_gt, ys_gt), xd_gt, yd_gt, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect_gt)
        # red pred box
        if 'coords' in locals():
            rect_p = patches.Rectangle((xs_p, ys_p), xd_p, yd_p, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect_p)
        plt.savefig(os.path.join(args.save_folder, 'test_'+str(idx)+'.png'))
        plt.close()



if __name__ == '__main__':
    # load net
    num_classes = 2 # 1 lesion + 1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    #testset = VOCDetection(args.voc_root, [('2007', 'test')], None, AnnotationTransform())
    """"########## Data Loading & dimension matching ##########"""
    # load custom CT dataset
    datapath = '/home/tkdrlf9202/Datasets/liver_lesion_aligned/lesion_dataset_4phase_aligned.h5'
    train_sets = [('liver_lesion')]


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

    # split train & valid set, subject-level (without shuffle)
    ct_train, ct_valid, coord_ssd_train, coord_ssd_valid = train_test_split(ct, coord, test_size=0.1, shuffle=False)

    # flatten the subject & sample dimension for each sets by stacking
    ct_train = np.vstack(ct_train)
    ct_valid = np.vstack(ct_valid)
    coord_ssd_train = np.vstack(coord_ssd_train).astype(np.float64)
    coord_ssd_valid = np.vstack(coord_ssd_valid).astype(np.float64)
    """#########################################################"""

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    means = (34, 34, 34)
    validset = FISHdetection(ct_valid, coord_ssd_valid, None, 'lesion_valid')
    allset = FISHdetection(np.vstack(ct), np.vstack(coord).astype(np.float64), None, 'lesion_all')

    test_net(args.save_folder, net, args.cuda, allset,
             BaseTransform(net.size, means),
             thresh=args.visual_threshold)