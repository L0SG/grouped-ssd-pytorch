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
import torch.utils.data as data
from ssd import build_ssd
import numpy as np
import h5py
from layers.box_utils import nms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='/home/tkdrlf9202/PycharmProjects/Liver_segmentation/ssd_liverdet/weights/ssd300_0712_5000.pth',
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
        # (y, x) start & delta of ground truth
        ys_gt, xs_gt = annotation[0][0], annotation[0][1]
        yd_gt, xd_gt = annotation[0][2] - ys_gt, annotation[0][3] - xs_gt
        # (y, x) start & delta of prediction
        ys_p, xs_p = int(coords[1]), int(coords[0])
        yd_p, xd_p = int(coords[3]) - ys_p, int(coords[2]) - xs_p

        # visualization: draw gt & predicted bounding box and save to image
        output_image = img.copy()
        fig, ax = plt.subplots(1)
        ax.imshow(output_image)
        # green gt box
        rect_gt = patches.Rectangle((xs_gt, ys_gt), xd_gt, yd_gt, linewidth=1, edgecolor='g', facecolor='none')
        # red pred box
        rect_p = patches.Rectangle((xs_p, ys_p), xd_p, yd_p, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect_gt)
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
    # ct: [len_data, 512, 512]
    # since base network receives rgb image, just make 2 more channels and fill the values in
    # this is temporary: using upper and lower slice instead will be better
    # make 0~155 uint8 image
    ct_rgb = np.zeros((ct.shape[0], 512, 512, 3), dtype=np.uint8)
    for idx in range(ct.shape[0]):
        ct_img = ct[idx]
        ct_rgb[idx, ..., 0] = ct_img * 255
        ct_rgb[idx, ..., 1] = ct_img * 255
        ct_rgb[idx, ..., 2] = ct_img * 255

    # coord: [len_data, 4] => should extend to 5 (include label)
    # all coords are lesion: add class label "1"
    coord_with_label = np.zeros((coord.shape[0], 5))
    for idx in range(coord.shape[0]):
        crd = coord[idx]
        # add zero as class index: it is treated to 1 by adding +1 to that
        # loss function automatically defines another zero as background
        # https://github.com/amdegroot/ssd.pytorch/issues/17
        crd = np.append(crd, [0])
        coord_with_label[idx] = crd
    """#########################################################"""

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    means = (34, 34, 34)
    testset = FISHdetection(ct_rgb, coord_with_label, None, 'fish_detection')
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, means),
             thresh=args.visual_threshold)
