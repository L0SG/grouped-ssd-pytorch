import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
#from data import v2, v1, AnnotationTransform, VOCDetection, detection_collate, VOCroot, VOC_CLASSES
from data import FISHdetection, detection_collate, v2, v1
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd_multiphase_custom_512 import build_ssd
import numpy as np
import time
import h5py
from sklearn.model_selection import train_test_split

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")



parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
# parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=True, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
# parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

#cfg = (v1, v2)[args.version == 'v2']

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

"""#################### Hyperparameters ####################"""
ssd_dim = 512
# current CT dataset has mean pixel val of 33.5
means = (34, 34, 34)
num_classes = 2 # lesion or background
batch_size = args.batch_size
#accum_batch_size = 32
#iter_size = accum_batch_size / batch_size
max_iter = 50000
weight_decay = 0.0005
stepvalues = (20000, 30000, 40000)
gamma = 0.1
momentum = 0.9
# use batchnorm for vgg
batch_norm = True

# data augmentation hyperparams
gt_pixel_jitter = 0.01
expand_ratio = 1.5
"""#########################################################"""


if args.visdom:
    import visdom
    viz = visdom.Visdom()


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

"""
# for debug data with one slice per subject
ct_train = (np.array(ct).transpose([0, 1, 3, 4, 2]) * 255).astype(np.uint8)
coord_ssd_train = np.array(coord).astype(np.float64)
"""
"""#########################################################"""

"""#################### Network Definition ####################"""
ssd_net = build_ssd('train', 512, num_classes, batch_norm=batch_norm)
net = ssd_net

if args.cuda:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
else:
    #vgg_weights = torch.load(args.save_folder + args.basenet)
    print('pretrained weights not loaded: training from scratch...')
    # print('Loading base network...')
if args.cuda:
    net = net.cuda()


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if not args.resume:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)
"""#########################################################"""


def train():
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    print('Loading Dataset...')
    dataset_train = FISHdetection(ct_train, coord_ssd_train,
                                  SSDAugmentation(gt_pixel_jitter, expand_ratio, ssd_dim, means), dataset_name='liver_lesion_train')
    dataset_valid = FISHdetection(ct_valid, coord_ssd_valid,
                                  SSDAugmentation(gt_pixel_jitter, expand_ratio, ssd_dim, means), dataset_name='liver_detection_valid')

    epoch_size = len(dataset_train) // args.batch_size
    print('Training SSD on', dataset_train.name)
    step_index = 0

    # visdom plot
    if args.visdom:
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current SSD_Liver Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
        valid_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='SSD_Liver Validation Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
        epoch_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Epoch SSD_Liver Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )

    batch_iterator = None
    data_loader_train = data.DataLoader(dataset_train, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    data_loader_valid = data.DataLoader(dataset_valid, batch_size, num_workers=args.num_workers,
                                        shuffle=True, collate_fn=detection_collate, pin_memory=True)

    for iteration in range(args.start_iter, max_iter):
        net.train()
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader_train)
        if iteration in stepvalues:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * epoch,
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu() / epoch_size,
                    win=epoch_lot,
                    update='append'
                )

            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = images.cuda().view(images.shape[0], -1, images.shape[3], images.shape[4])
            images = Variable(images)
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = images.view(images.shape[0], -1, images.shape[3], images.shape[4])
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]

        """ DEBUG CODE: printout augmented images & targets"""
        if False:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from PIL import Image
            print('Debug mode: printing augmented data...')
            images_print = images.data[:, :, :, :].cpu().numpy()
            images_print[images_print < 0] = 0
            targets_print = np.array([target.data[0].cpu().numpy().squeeze()[:4] for target in targets])
            targets_print *= images_print.shape[2]
            images_print = images_print.astype(np.uint8)

            # center format to min-max format
            min_x, min_y, max_x, max_y = targets_print[:, 0], targets_print[:, 1], targets_print[:, 2], targets_print[:, 3]
            width = (max_x - min_x).astype(np.int32)
            height = (max_y - min_y).astype(np.int32)
            min_x = min_x.astype(np.int32)
            min_y = min_y.astype(np.int32)

            for idx in range(images_print.shape[0]):
                for idx_img in range(images_print.shape[1]):
                    # visualization: draw gt & predicted bounding box and save to image
                    output_image = images_print[idx, idx_img]
                    fig, ax = plt.subplots(1)
                    ax.imshow(output_image, cmap='gray')
                    # green gt box
                    rect_gt = patches.Rectangle((min_x[idx], min_y[idx]), width[idx], height[idx], linewidth=1, edgecolor='g', facecolor='none')
                    ax.add_patch(rect_gt)
                    plt.savefig(os.path.join('debug', 'train_' + str(idx) + '_' + str(idx_img) + '.png'))
                    plt.close()
            exit()

        # forward
        t0 = time.time()
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]

        # train log
        if iteration % 10 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                viz.image(images.data[random_batch_index].cpu().numpy())

        # validation phase for each several train iter
        if iteration % 100 == 0 and iteration > 10:
            del images, targets
            net.eval()
            loss_l_val, loss_c_val, loss_val = 0., 0., 0.
            batch_iterator_val = iter(data_loader_valid)
            for idx in range(len(batch_iterator_val)):
                img_val, tar_val = next(batch_iterator_val)
                if args.cuda:
                    img_val = img_val.cuda().view(img_val.shape[0], -1, img_val.shape[3], img_val.shape[4])
                    img_val = Variable(img_val, volatile=True)
                    tar_val = [Variable(anno.cuda(), volatile=True) for anno in tar_val]
                else:
                    img_val = img_val.view(img_val.shape[0], -1, img_val.shape[3], img_val.shape[4])
                    img_val = Variable(img_val, volatile=True)
                    tar_val = [Variable(anno, volatile=True) for anno in tar_val]

                out_val = net(img_val)
                loss_l_val_step, loss_c_val_step = criterion(out_val, tar_val)
                loss_val_step = loss_l_val_step + loss_c_val_step
                loss_l_val += loss_l_val_step
                loss_c_val += loss_c_val_step
                loss_val += loss_val_step
                del out_val
            loss_l_val, loss_c_val, loss_val = loss_l_val/(idx+1), loss_c_val/(idx+1), loss_val/(idx+1)
            print('\n')
            print('VALID: iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss_val.data[0]), end='\n')
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * iteration,
                    Y=torch.Tensor([loss_l_val.data[0], loss_c_val.data[0],
                                    loss_l_val.data[0] + loss_c_val.data[0]]).unsqueeze(0).cpu(),
                    win=valid_lot,
                    update='append'
                )
            del img_val, tar_val

        # visdom train plot
        # skip the first 10 iteration plot: too high loss, less pretty
        if iteration > 10:
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * iteration,
                    Y=torch.Tensor([loss_l.data[0], loss_c.data[0],
                        loss_l.data[0] + loss_c.data[0]]).unsqueeze(0).cpu(),
                    win=lot,
                    update='append'
                )
                # hacky fencepost solution for 0th epoch plot
                if iteration == 0:
                    viz.line(
                        X=torch.zeros((1, 3)).cpu(),
                        Y=torch.Tensor([loc_loss, conf_loss,
                            loc_loss + conf_loss]).unsqueeze(0).cpu(),
                        win=epoch_lot,
                        update=True
                    )

        # save checkpoint
        if iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_vgggroup_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(), args.save_folder + '' + args.version + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
