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
from sklearn.model_selection import train_test_split, KFold
from average_precision import APCalculator
from collections import defaultdict, namedtuple
import pickle

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_allgroup_v2custom_BN_CV0_9000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--visual_threshold', default=0.01, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
#parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def test_net(save_folder, net, cuda, testset, transform, top_k,
             imsize=300, thresh=0.05):
    num_images = len(testset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(2)]
    output_dir = get_output_dir('ssd300_120000', 'test')
    det_file = os.path.join(output_dir, 'detections.pkl')
    """
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test1.txt'
    num_images = len(testset)
    """
    """
    # define ap calculator
    ap_calculator = APCalculator()
    
    # define box namedtuple for the calculator
    Box = namedtuple('Box', ['label', 'labelid', 'coords', 'size'])
    """


    # define ground truth and prediciton list
    # length is the number of valid set images
    # elements of predictions contains [img ID, confidence, coords]
    ground_truth = []
    predictions = []
    # class_recs extract gt and detected flag for AP calculation
    class_recs = {}
    # total number of positive gt boxes
    npos = 0

    for idx in range(num_images):
        print('Testing image {:d}/{:d}....'.format(idx+1, num_images))
        # pull img & annotations
        img = testset.pull_image(idx)
        annotation = [testset.pull_anno(idx)]
        # base transform the image and permute to [phase, channel, h, w] and flatten to [1, channel, h, w]
        x = torch.from_numpy(transform(img)[0]).permute(0, 3, 1, 2).contiguous()
        x = x.view(-1, x.shape[2], x.shape[3])
        x = Variable(x, volatile=True).unsqueeze(0)
        if cuda:
            x = x.cuda()
        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[2], img.shape[1],
                             img.shape[2], img.shape[1]])

        # mask out detections with zero confidence
        detections = detections[0, 1, :]
        mask = detections[:, 0].gt(0.).expand(5, detections.size(0)).t()
        detections = torch.masked_select(detections, mask).view(-1, 5)
        if detections.dim() == 0:
            continue

        # we only care for lesion class (1 in dim=1)
        score = detections[:, 0].cpu().numpy()
        coords = (detections[:, 1:] * scale).cpu().numpy()
        boxes = np.hstack((np.expand_dims(score, 1), coords))
        # attach img id to the leftmost side
        img_id = np.ones((boxes.shape[0], 1)) * idx
        boxes = np.hstack((img_id, boxes))

        # append ground truth and extend the predictions
        # cut out the label (last elem) since it's not necessary for AP
        ground_truth.append(annotation[0][:, :-1])
        predictions.extend(boxes)

        # only use the portal phase bbox
        bbox = np.expand_dims(annotation[0][2, :-1], 0)
        det = [False]
        npos += 1
        class_recs[idx] = {'bbox': bbox,
                            'det': det}
        # bbox = annotation[0][:, :-1]
        # det = [False] * bbox.shape[0]
        # npos += bbox.shape[0]



    print('test')

    # sort the prediction in descending global confidence
    # first parse predictions into img ids, confidence and bb
    image_ids = [x[0] for x in predictions]
    confidence = np.array([float(x[1]) for x in predictions])
    BB = np.array([[float(z) for z in x[2:]] for x in predictions])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [int(image_ids[x]) for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                   (BBGT[:, 2] - BBGT[:, 0]) *
                   (BBGT[:, 3] - BBGT[:, 1]) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > 0.5:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric=True)

    print('test')




if __name__ == '__main__':
    """"########## Data Loading & dimension matching ##########"""
    # load custom CT dataset
    datapath = '/home/tkdrlf9202/Datasets/liver_lesion_aligned/lesion_dataset_4phase_aligned.h5'
    train_sets = [('liver_lesion')]
    cross_validation = 5
    cv_idx_for_test = 0

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

    test_net(args.save_folder, net, args.cuda, trainset,
             BaseTransform(size, means), args.top_k, size,
             thresh=args.visual_threshold)