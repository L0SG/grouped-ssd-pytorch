import os
import torch
import numpy as np
torch.manual_seed(1111)
np.random.seed(1111)
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
#from data import v2, v1, AnnotationTransform, VOCDetection, detection_collate, VOCroot, VOC_CLASSES
from data import DataSplitter, FISHdetectionV2, detection_collate_v2, BaseTransform
from utils.augmentations import SSDAugmentation
from utils.check_grad_norm import check_grad_norm
from layers.modules import MultiBoxLoss
from tqdm import tqdm

import time
import copy
from test_ap_iobb import test_net
import tensorboardX
from torchvision import utils as vutils
import cv2

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--datapath', default='/media/hdd/tkdrlf9202/Datasets/liver_year1_dataset_extended_1904_preprocessed/ml_ready', type=str, help='data path')
parser.add_argument('--load_data_to_ram', default=False, type=str2bool)
parser.add_argument('--ssd_type', default=None, type=str, choices=['gssd', 'ssd', 'fssd', 'fusedssd'])

parser.add_argument('--p_only', default=False, type=str2bool, help='only use portal phase ct image.')

parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--num_classes', default=2, type=int, help='lesion or background, keep 2')
parser.add_argument('--ssd_dim', default=300, type=int, help='ssd dim. keep 300')
parser.add_argument('--means', type=float, default=49., help='mean pixel value of data')

# custom ssd-specific tuners
parser.add_argument('--groups_vgg', default=4, type=int)
parser.add_argument('--groups_extra', default=4, type=int)
parser.add_argument('--feature_scale', default=1, type=int)
parser.add_argument('--use_fuseconv', default=True, type=str2bool)
parser.add_argument('--use_self_attention', default=False, type=str2bool, help='use self-attention in SAGAN before each source points')
parser.add_argument('--use_self_attention_base', default=False, type=str2bool, help='use self-attention inside the vgg+extra base backbone cnn')
parser.add_argument('--num_dcn_layers', default=0, type=int, help='use deformable conv (DCNv2) before each source points, defaults to 0 (do not use)')
parser.add_argument('--groups_dcn', default=1, type=int)
parser.add_argument('--dcn_cat_sab', default=False, type=str2bool, help='concat feature map & self-attention map before feeding to dcn. requires --use_self_attention_base to True')
parser.add_argument('--detach_sab', default=False, type=str2bool, help='detach sab attention map concanated with x before feeding it to dcn. requires --dcn_cab_sab to True')

# self-attn specific tuners
parser.add_argument('--max_pool_factor', default=1, type=int, help="factor used for maxpool op inside Self_attn. ex: 2 uses 2x downsampling for self att map")

parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')

parser.add_argument('--max_iter', default=10001, type=int, help='total training iterations')
parser.add_argument('--val_every', default=500, type=int, help='validate & test the model interval')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float, help='initial learning rate')

parser.add_argument('--modify_dcn_lr', default=False, type=str2bool,
                    help='use x0.1 learning rate for dcn layers')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--stepvalues', nargs="*", type=int, default=[5000, 8000], help='number of iterations for StepLR')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--batch_norm', default=True, type=str2bool, help='wheter to use batch norm')
parser.add_argument('--clip', default=None, type=float, help='whether to clip grad_norm given norm value. no clip if None.')

parser.add_argument('--ohnm_neg_ratio', default=1, type=int, help='OHNM (online hard neg mining) ratio (pos:neg = 1:x)')

# data augmentation
parser.add_argument('--gt_pixel_jitter', default=0.01, type=float)
parser.add_argument('--expand_ratio', default=1.5, type=float)

parser.add_argument('--cross_validation', default=1, type=int, help='cross validation')

parser.add_argument('--log_iters', default=True, type=str2bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')


parser.add_argument('--output', default="/home/tkdrlf9202/Experiments/gssd", help='output folder')
parser.add_argument('--model_name', default='gssd', help='model name for this run')
# parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')

# model testing parameter
parser.add_argument('--test', default=False, type=str2bool, help='run testing parat onluy')
parser.add_argument('--checkpoint', default=None, type=int, help='checkpoint loading. specify iteration.')
parser.add_argument('--confidence_threshold', default=0.01, type=float, help='for ap calculation')
parser.add_argument('--use_07_metric', default=True, type=str2bool, help='if False, uses 11 metric')
parser.add_argument('--ap_list', default='0.5', type=str, help='if "0.1,0.5", it means AP10 and AP50')
parser.add_argument('--iobb_list', default='0.5', type=str, help='if "0.1,0.5", it means IOBB10 and IOBB50')

# visualize mode of model behavior
parser.add_argument('--visualize', default=False, type=str2bool, help='visualize model behavior.')

# poc switch
parser.add_argument('--aug_method', default='vanilla', choices=['vanilla', 'cuda'])
parser.add_argument('--use_normalize', default=False, type=str2bool, help='normalize to [0, 1] before feeding to model')
parser.add_argument('--speedrun', default=1, type=int, help='speedrun. skip evaluation & ap calculation up to the specified iteration (since this is a sluggish cpu-bound one)')

args = parser.parse_args()
# auto-adjust means
args.means = [args.means] * 3

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def build_ssd_wrapper(phase, args):
    # simple wrapper that takes args, then select network with given args accordingly
    print("loading {} model".format(args.ssd_type))
    if args.ssd_type == 'gssd':
        from models.ssd_multiphase_custom_group import build_ssd
    # elif args.ssd_type == 'ssd':
    #     from ssd_multiphase_custom import build_ssd
    # elif args.ssd_type == 'fssd':
    #     from ssd_multiphase_custom_fssd import build_ssd
    # elif args.ssd_type == 'fusedssd':
    #     from ssd_multiphase_custom_fused import build_ssd
    else:
        raise NotImplementedError("unkown ssd_type")

    GROUPS_VGG = args.groups_vgg
    GROUPS_EXTRA = args.groups_extra
    feature_scale = args.feature_scale
    use_fuseconv = args.use_fuseconv

    ssd_net = build_ssd(phase, args.ssd_dim, args.num_classes, args.batch_norm,
                        args.groups_vgg, args.groups_extra, args.feature_scale, args.use_fuseconv,
                        args.use_self_attention, args.use_self_attention_base, args.num_dcn_layers, args.groups_dcn, args.dcn_cat_sab, args.detach_sab,
                        args.max_pool_factor)

    return ssd_net


def train():
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0

    # hard-define the epoch size to the minimum of set in CV: minimal impact
    epoch_size = []
    for idx in range(args.cross_validation):
        epoch_size.append(len(cv_train[idx]))
    epoch_size = min(epoch_size)

    step_index = 0

    batch_iterator = None

    for iteration in range(args.start_iter, args.max_iter):
        for idx in range(args.cross_validation):
            net_cv[idx].train()
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = []
            for idx in range(args.cross_validation):
                batch_iterator.append(iter(cv_train[idx]))
        if iteration in args.stepvalues:
            step_index += 1
            for idx in range(args.cross_validation):
                adjust_learning_rate(optimizer_cv[idx], args.gamma, step_index)
            epoch += 1
        """
        if iteration == 2000:
            for idx in range(cross_validation):
                # Freeze the conf layers after some iters to prevent overfitting
                for conf_param in net_cv[idx].module.conf.parameters():
                    conf_param.requires_grad = False
        """
        # load train data
        loss_cv = 0.
        loc_loss_cv = 0.
        conf_loss_cv = 0.
        grad_norm_cv = 0.

        t0_full = time.time()
        t_cv = 0
        for idx in range(args.cross_validation):
            images, targets = next(batch_iterator[idx])

            if args.cuda:
                images = images.cuda().view(images.shape[0], -1, images.shape[3], images.shape[4])
                images = Variable(images)
                targets = [Variable(anno.cuda()) for anno in targets]
            else:
                images = images.view(images.shape[0], -1, images.shape[3], images.shape[4])
                images = Variable(images)
                targets = [Variable(anno) for anno in targets]
                targets.requires_grad = False
            for i in range(len(targets)):
                targets[i].requires_grad = False

            """ DEBUG CODE: printout augmented images & targets"""
            if False:
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
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
            out = net_cv[idx](images)

            # backprop
            optimizer_cv[idx].zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            # check grad_norm
            grad_norm = check_grad_norm(net_cv[idx])
            if args.clip is not None:
                grad_norm = nn.utils.clip_grad_norm_(net_cv[idx].parameters(), args.clip)
            optimizer_cv[idx].step()
            t1 = time.time()

            t_cv += (t1 - t0)
            # loss_cv += loss.data[0]
            # loc_loss_cv += loss_l.data[0]
            # conf_loss_cv += loss_c.data[0]
            loss_cv += loss.item()
            loc_loss_cv += loss_l.item()
            conf_loss_cv += loss_c.item()
            grad_norm_cv += grad_norm
            del out

        loss_cv, loc_loss_cv, conf_loss_cv, grad_norm_cv = \
            loss_cv / args.cross_validation, loc_loss_cv / args.cross_validation, conf_loss_cv / args.cross_validation, grad_norm_cv / args.cross_validation

        t1_full = time.time()
        # train log
        if iteration % 1 == 0:
            print('Timer: {:.4f} sec, {:.4f} full sec'.format(t_cv, (t1_full - t0_full)))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss_cv), end=' ')

            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                viz.image(images.data[random_batch_index].cpu().numpy())

            # write text log
            f_train.write(str(iteration)+'\t'+str(loss_cv)+'\t'+str(loc_loss_cv)+'\t'+str(conf_loss_cv)+'\n')
            f_train.flush()

            # visdom train plot
            # skip the first 10 iteration plot: too high loss, less pretty
            if iteration > 10:
                writer.add_scalar('loss/loss_cv', loss_cv, iteration)
                writer.add_scalar('loss/loc_loss_cv', loc_loss_cv, iteration)
                writer.add_scalar('loss/conf_loss_cv', conf_loss_cv, iteration)
                writer.add_scalar('meta/grad_norm', grad_norm_cv, iteration)
                # get lr for sanity check
                lr_cur = optimizer_cv[0].param_groups[0]['lr']
                writer.add_scalar('meta/lr', lr_cur, iteration)

        if iteration % 100 == 0:
            random_batch_index = np.random.randint(images.size(0))
            input_visual = get_input_visual(images, targets, random_batch_index, args.use_normalize)
            writer.add_image('input', input_visual, iteration)
            writer.flush()

        # validation phase for each several train iter
        if iteration % args.val_every == 0 and iteration > 1 and iteration > args.speedrun:
        # if iteration % args.val_every == 0:
            del images, targets
            for idx in range(args.cross_validation):
                net_cv[idx].eval()
            loss_l_val, loss_c_val, loss_val = 0., 0., 0.

            eval_iter_counter = 0
            for idx_c in range(args.cross_validation):
                batch_iterator_val = iter(cv_valid[idx_c])
                for idx in tqdm(range(len(batch_iterator_val))):
                    img_val, tar_val = next(batch_iterator_val)
                    if args.cuda:
                        img_val = img_val.cuda().view(img_val.shape[0], -1, img_val.shape[3], img_val.shape[4])
                        img_val = Variable(img_val)
                        tar_val = [Variable(anno.cuda()) for anno in tar_val]
                    else:
                        img_val = img_val.view(img_val.shape[0], -1, img_val.shape[3], img_val.shape[4])
                        img_val = Variable(img_val)
                        tar_val = [Variable(anno) for anno in tar_val]
                    with torch.no_grad():
                        out_val = net_cv[idx_c](img_val)
                    loss_l_val_step, loss_c_val_step = criterion(out_val, tar_val)
                    loss_val_step = loss_l_val_step + loss_c_val_step
                    loss_l_val += loss_l_val_step.item()
                    loss_c_val += loss_c_val_step.item()
                    loss_val += loss_val_step.item()
                    del out_val
                    eval_iter_counter += 1
            for idx in range(args.cross_validation):
                net_cv[idx].train()

            # loss_l_val, loss_c_val, loss_val = \
            #     loss_l_val/eval_iter_counter,\
            #     loss_c_val/eval_iter_counter,\
            #     loss_val/eval_iter_counter

            ap, iobb, ap_test, iobb_test = test_net_wrapper(iteration)

            print('average AP for valid set: ' + " ".join(str(a) for a in ap))
            print('average IoBB for valid set: ' + " ".join(str(a) for a in iobb))
            print('VALID: iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss_val), end='\n')

            writer.add_scalar('eval_loss/loss_l_val', loss_l_val, iteration)
            writer.add_scalar('eval_loss/loss_c_val', loss_c_val, iteration)
            writer.add_scalar('eval_loss/loss_val', loss_val, iteration)
            for i, a in enumerate(ap):
                writer.add_scalar('eval_ap/ap'+str(ap_list[i]), a, iteration)
            for i, a in enumerate(iobb):
                writer.add_scalar('eval_ap/iobb'+str(iobb_list[i]), a, iteration)
            for i, a in enumerate(ap_test):
                writer.add_scalar('test_ap/ap'+str(ap_list[i]), a, iteration)
            for i, a in enumerate(iobb_test):
                writer.add_scalar('test_ap/iobb'+str(ap_list[i]), a, iteration)

            del img_val, tar_val
            # write valid log
            write_string = str(iteration) + '\t' + str(loss_val) + '\t' + str(loss_l_val) + '\t' + str(loss_c_val) + '\t'
            write_string_test = str(iteration) + '\t'
            for a in ap:
                write_string += (str(a) + '\t')
            for a in iobb:
                write_string += (str(a) + '\t')
            for a in ap_test:
                write_string_test += (str(a) + '\t')
            for a in iobb_test:
                write_string_test += (str(a) + '\t')
            write_string += '\n'
            write_string_test += '\n'
            f_valid.write(write_string)
            f_valid.flush()
            f_test.write(write_string_test)
            f_test.flush()
            writer.flush()

        # save checkpoint
        if iteration % args.val_every == 0:
            print('Saving state, iter:', iteration)
            for idx in range(args.cross_validation):
                state = {'iters': iteration,
                         'state_dict': net_cv[idx].state_dict(),
                         'optimizer': optimizer_cv[idx].state_dict()}
                torch.save(state, os.path.join(output_path, "checkpoints", args.model_name, args.model_name + '_' + str(iteration) + '_CV' +
                           str(idx) + '.pth'))
    # torch.save(net[idx].state_dict(), args.save_folder + '' + args.version + '.pth')


def test_net_wrapper(iteration):
    # calculate AP
    print('\n')
    ap = [0.]*len(ap_list)
    iobb = [0.]*len(iobb_list)

    ap_test = [0.]*len(ap_list)
    iobb_test = [0.]*len(iobb_list)
    for idx in range(args.cross_validation):
        # load weights trained with Dataparallel
        # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
        state_dict = net_cv[idx].state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        net_ap = build_ssd_wrapper('test', args).cuda()
        # net_ap = build_ssd('test', args.ssd_dim, args.num_classes, batch_norm=args.batch_norm).cuda()
        net_ap.load_state_dict(new_state_dict)
        net_ap.eval()
        new_ap, new_iobb = test_net(net_ap, args.cuda, cv_ap[idx], base_aug, args.ssd_dim, thresh=args.confidence_threshold,
                                    mode='v2', use_07_metric=args.use_07_metric, ap_list=ap_list, iobb_list=iobb_list,
                                    writer=writer, iteration=iteration, visualize=args.visualize, output_path=args.output, model_name=args.model_name)
        new_ap_test, new_iobb_test = test_net(net_ap, args.cuda, testset_ap, base_aug, args.ssd_dim, thresh=args.confidence_threshold,
                                              mode='v2', use_07_metric=args.use_07_metric, ap_list=ap_list, iobb_list=iobb_list,
                                              writer=writer, iteration=iteration, visualize=args.visualize, output_path=args.output, model_name=args.model_name)
        if args.visualize:
            continue  # skip measuring

        ap = [ap[i]+a for i, a in enumerate(new_ap)]
        iobb = [iobb[i] + a for i, a in enumerate(new_iobb)]
        ap_test = [ap_test[i]+a for i, a in enumerate(new_ap_test)]
        iobb_test = [iobb_test[i]+a for i, a in enumerate(new_iobb_test)]
        del net_ap

    if args.visualize:
        return None, None, None, None

    ap = [a/args.cross_validation for a in ap]
    iobb = [a / args.cross_validation for a in iobb]
    ap_test = [a/args.cross_validation for a in ap_test]
    iobb_test = [a/args.cross_validation for a in iobb_test]
    return ap, iobb, ap_test, iobb_test


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_input_visual(images, targets, random_batch_index, use_normalize):
    # consider single image as [4, 3, H, W], then take the middle, [4, 1, H, W]
    image_for_vis = images[random_batch_index].clone().cpu()
    image_for_vis = image_for_vis.view((4, 3, image_for_vis.size(1), image_for_vis.size(2)))
    image_for_vis = image_for_vis[:, 1, :, :].unsqueeze(1)
    img_out = []
    target_for_vis = targets[random_batch_index].cpu()
    for i in range(image_for_vis.shape[0]):
        img_rgb = np.repeat(image_for_vis[i, :, :], 3, axis=0).permute(1, 2, 0).numpy().copy()
        if not use_normalize:
            # then force-normalize here for proper vis
            img_min, img_max = img_rgb.min(), img_rgb.max()
            img_rgb = (img_rgb - img_min) / (img_max - img_min)
        for j in range(target_for_vis.shape[0]):
            xmin, ymin, xmax, ymax = (target_for_vis[j][:4] * args.ssd_dim).long().numpy()
            cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), (1, 0, 0), 1)
        img_out.append(torch.tensor(img_rgb))
    img_out = torch.stack(img_out).permute(0, 3, 1, 2)

    input_visual = vutils.make_grid(img_out)
    return input_visual

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    output_path = args.output
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "logs", args.model_name), exist_ok=True)
    os.makedirs(os.path.join(output_path, "checkpoints", args.model_name), exist_ok=True)

    ap_list = args.ap_list.split(',')
    ap_list = [float(a) for a in ap_list]
    iobb_list = args.iobb_list.split(',')
    iobb_list = [float(a) for a in iobb_list]

    if not args.test:
        # create train & valid log text file
        f_train = open(os.path.join(output_path, "logs", args.model_name, "train_log" + '.txt'), 'w+')
        f_train.write('iteration\tloss\tloc_loss\tconf_loss\n')
        f_valid = open(os.path.join(output_path, "logs", args.model_name, "valid_log" + '.txt'), 'w+')
        f_test = open(os.path.join(output_path, "logs", args.model_name, "test_log" + ".txt"), 'w+')
        write_string = 'iteration\tloss\tloc_loss\tconf_loss\t'
        write_string_test = 'iteration\t'
        for a in ap_list:
            write_string += 'AP'+str(a)+'\t'
            write_string_test += 'AP'+str(a)+'\t'
        for a in iobb_list:
            write_string += 'IoBB'+str(a)+'\t'
            write_string_test += 'IoBB'+str(a)+'\t'
        f_valid.write(write_string+'\n')
        f_test.write(write_string_test+'\n')

        f_args = open(os.path.join(output_path, "logs", args.model_name, "args_log" + ".txt"), 'w+')

    for item in vars(args):
        key, val = item, getattr(args, item)
        print(key, val)
        if not args.test:
            f_args.write(str(key) + ' ' + str(val))
            f_args.write('\n')
    if not args.test:
        f_args.write('\n')
        f_args.close()

    writer = tensorboardX.SummaryWriter(log_dir=os.path.join(output_path, "logs", args.model_name))

    """"########## Data Loading & dimension matching ##########"""

    data_splitter = DataSplitter(args.datapath, cross_validation=args.cross_validation, num_test_subject=10)

    cv_train = []
    cv_valid = []
    cv_ap = []

    if args.aug_method == "vanilla":
        ssd_aug = SSDAugmentation(args.gt_pixel_jitter, args.expand_ratio, args.ssd_dim, args.means,
                                  use_normalize=args.use_normalize, p_only=args.p_only)
    elif args.aug_method == "cuda":
        raise NotImplementedError("--aug_method = 'cuda' is deprecated!")
        # ssd_aug = SSDAugmentationCUDA(args.gt_pixel_jitter, args.expand_ratio, args.ssd_dim, args.means,
        #                               use_normalize=args.use_normalize, p_only=args.p_only)
    # for ap calculation
    base_aug = BaseTransform(args.ssd_dim, args.means, use_normalize=args.use_normalize, p_only=args.p_only)

    for i in range(args.cross_validation):
        cv_train.append(data.DataLoader(FISHdetectionV2(args.datapath, data_splitter.data_cv_train[i],
                                                        ssd_aug,
                                                        dataset_name='lesion_cv_train_' + str(i),
                                                        load_data_to_ram=args.load_data_to_ram),
                                        batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, collate_fn=detection_collate_v2, drop_last=True,
                                        pin_memory=False))
        cv_valid.append(data.DataLoader(FISHdetectionV2(args.datapath, data_splitter.data_cv_eval[i],
                                                        ssd_aug,
                                                        dataset_name='lesion_cv_valid_' + str(i),
                                                        load_data_to_ram=args.load_data_to_ram),
                                        batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, collate_fn=detection_collate_v2, drop_last=False,
                                        pin_memory=False))
        # ap dataset is not wrapped with dataloader
        cv_ap.append(FISHdetectionV2(args.datapath, data_splitter.data_cv_eval[i],
                                     None,
                                     dataset_name='lesion_cv_ap_' + str(i), load_data_to_ram=args.load_data_to_ram))

    testset_ap = FISHdetectionV2(args.datapath, data_splitter.data_test,
                                 None,
                                 dataset_name='lesion_test_ap', load_data_to_ram=args.load_data_to_ram)

    if args.cross_validation > 1:
        print('using {}-fold CV...'.format(args.cross_validation))
    elif args.cross_validation == 1:
        print('using single train-valid split...')
    for idx in range(args.cross_validation):
        print(len(cv_train[idx]), len(cv_valid[idx]))

    """#################### Network Definition ####################"""
    net = build_ssd_wrapper('train', args)
    print("model parameters: {}".format(count_parameters(net)))


    def xavier(param):
        init.xavier_uniform_(param)


    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            xavier(m.weight.data)
            m.bias.data.zero_()


    # if args.resume:
    #     print('Resuming training, loading {}...'.format(args.resume))
    #     net.load_weights(args.resume)

    # else:
        # vgg_weights = torch.load(args.save_folder + args.basenet)
        # print('pretrained weights not loaded: training from scratch...')
        # print('Loading base network...')
        # initialize newly added layers' weights with xavier method
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # for 5-fold CV, define 5 copies of the model and optimizer
    print('copying models & optimizers for CV...')
    net_cv = []
    optimizer_cv = []
    for idx in range(args.cross_validation):
        net_cv.append(copy.deepcopy(net))

        if args.num_dcn_layers > 0 and args.modify_dcn_lr:
            print("INFO: modifying learning rate for DCN by x0.1")
            # https://discuss.pytorch.org/t/how-to-set-a-different-learning-rate-for-a-single-layer-in-a-network/48552/11
            if hasattr(net_cv[idx], 'module'): # DP case
                list_params_dcn = list(map(lambda x:x[1],
                                           list(filter(lambda kv: kv[0].startswith('module.dcn_list'), net_cv[idx].named_parameters()))))
                list_params_not_dcn = list(map(lambda x: x[1],
                                               list(filter(lambda kv: not kv[0].startswith('module.dcn_list'), net_cv[idx].named_parameters()))))
            else: # single gpu case
                list_params_dcn = list(map(lambda x:x[1],
                                           list(filter(lambda kv: kv[0].startswith('dcn_list'), net_cv[idx].named_parameters()))))
                list_params_not_dcn = list(map(lambda x: x[1],
                                               list(filter(lambda kv: not kv[0].startswith('dcn_list'), net_cv[idx].named_parameters()))))
            opt = optim.SGD(
                [
                    {"params": list_params_not_dcn},
                    {"params": list_params_dcn, "lr": args.lr * 0.1}
                ],
                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
            )

        else:
            opt = optim.SGD(net_cv[idx].parameters(), lr=args.lr,
                                          momentum=args.momentum, weight_decay=args.weight_decay)

        optimizer_cv.append(opt)

        if args.resume:
            print('Resuming training, loading {}...'.format(args.resume))
            # net_cv[idx] = torch.nn.DataParallel(net_cv[idx].cuda())
            checkpoint = torch.load(args.resume.replace('CV', 'CV{}'.format(idx)))
            net_cv[idx].load_state_dict(checkpoint['state_dict'])
            optimizer_cv[idx].load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            args.start_iter = checkpoint['iters']+1

    criterion = MultiBoxLoss(args.num_classes, 0.5, True, 0, True, args.ohnm_neg_ratio, 0.5, False, args.cuda)
    del net

    if args.checkpoint is not None:
        print("--checkpoint specified. loading checkpoint {}".format(args.checkpoint))
        for idx in range(args.cross_validation):
            checkpoint_path = os.path.join(args.output, "checkpoints", args.model_name,
                                           args.model_name + "_" + str(args.checkpoint) + "_CV" + str(idx) + ".pth")
            net_cv[idx].module.load_weights(checkpoint_path)

    """#########################################################"""
    if args.test:
        print("running test only...")
        f_test = open(os.path.join(output_path, "logs", args.model_name, "test_log" + ".txt"), 'a')
        # note: this does not actually sets the iteration variable for training.
        iteration = args.checkpoint
        ap, iobb, ap_test, iobb_test = test_net_wrapper(iteration)
        if args.visualize:
            print("visualization dump created at visualization subfolder.")
            exit()
        print_string = "AP valid: thresh {} 07_metric {} ".format(args.confidence_threshold, args.use_07_metric)
        for i, a in enumerate(ap):
            print_string += "{}: {} ".format(ap_list[i], a)
        print_string += "IoBB valid: thresh {} ".format(args.confidence_threshold)
        for i, a in enumerate(iobb):
            print_string += "{}: {} ".format(iobb_list[i], a)
        print(print_string)
        f_test.write(print_string + '\n')
        exit()
    else:
        train()
