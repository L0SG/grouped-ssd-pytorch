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
from ssd import build_ssd
import numpy as np
import h5py
from layers.box_utils import nms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='/home/tkdrlf9202/PycharmProjects/Liver_segmentation/ssd_liverdet/weights/ssd300_0712_105000.pth',
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
        img = testset.pull_image(idx)
        annotation = [testset.pull_anno(idx)]
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

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
    datapath = '/home/tkdrlf9202/Datasets/liver_lesion/lesion_dataset_Ponly_1332.h5'
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
                ct_flattened = dataset_h5['ct'][:]
                coordinate_flattened = dataset_h5['coordinate'][:]
                dataset_h5.close()

        return ct_flattened, coordinate_flattened


    ct, coord = load_lesion_dataset(datapath)
    # ct: [len_data, 3, 512, 512] (3 continous slides)
    # make channels last & 0~255 uint8 image
    ct = np.transpose(ct * 255, [0, 2, 3, 1]).astype(dtype=np.uint8)

    # coord: [len_data, 3, 4] => should extend to 5 (include label)
    # all coords are lesion: add class label "1"
    coord_ssd = np.zeros((coord.shape[0], 5))
    for idx in range(coord.shape[0]):
        # each channels have different gt => since they are nearly same, just use the middle gt as main target
        crd = coord[idx][1]
        # add zero as class index: it is treated to 1 by adding +1 to that
        # loss function automatically defines another zero as background
        # https://github.com/amdegroot/ssd.pytorch/issues/17
        crd = np.append(crd, [0])
        coord_ssd[idx] = crd

    # split train & valid set: subject-level (without shuffle)
    ct_train, ct_valid, coord_ssd_train, coord_ssd_valid = train_test_split(ct, coord_ssd, test_size=0.1, shuffle=False)
    """#########################################################"""

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    means = (34, 34, 34)
    validset = FISHdetection(ct_valid, coord_ssd_valid, None, 'lesion_valid')
    allset = FISHdetection(ct, coord_ssd, None, 'lesion_all')

    test_net(args.save_folder, net, args.cuda, allset,
             BaseTransform(net.size, means),
             thresh=args.visual_threshold)