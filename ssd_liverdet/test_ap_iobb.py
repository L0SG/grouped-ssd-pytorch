from __future__ import print_function
from torch.autograd import Variable
#from data import VOCroot, VOC_CLASSES as labelmap
# from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import copy
from tqdm import tqdm
from pixel_link.model import *
from pixel_link.postprocess import *

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


def prepare_background_img(img, annotation, boxes, gt_phase):
    # make rgb uint8
    annotation_ = [ano[0][:-1] for ano in annotation]
    img_ = np.repeat(np.expand_dims(img[:, :, :, 1], axis=-1), 3, axis=3)

    img_cat = []
    # draw annotation rectangle
    for i_p in range(img_.shape[0]):
        img_p = img_[i_p]
        for i_a in range(len(annotation_)):
            x_start, y_start, x_end, y_end = annotation_[i_a]
            cv2.rectangle(img_p, (x_start, y_start), (x_end, y_end), (255, 255, 0), 2)
            img_cat.append(img_p)
    # draw model prediction rectangle to gt phase img
    img_det = img_cat[gt_phase].copy()
    for i_d in range(len(boxes)):
        confidence = boxes[i_d][0]
        x_start, y_start, x_end, y_end = boxes[i_d][1:].astype(np.int)
        r, g, b = 255, int(255 * (1-confidence)), int(255 * (1-confidence))
        cv2.rectangle(img_det, (x_start, y_start), (x_end, y_end), (r, g, b), 2)

    img_cat = np.concatenate(img_cat, axis=1)

    return img_cat, img_det


def make_pred(net, cuda, testset, transform, thresh=0.05, mode='v1', writer=None, iteration=None,
              visualize=False, output_path=None, model_name=None, use_pixel_link=False):
    num_images = len(testset)

    # define ground truth and prediciton list
    # length is the number of valid set images
    # elements of predictions contains [img ID, confidence, coords]
    predictions = []
    # class_recs extract gt and detected flag for AP calculation
    class_recs = {}
    # total number of positive gt boxes
    npos = 0

    # temporary list that holds scores for tensorboard histogram
    score_nofilter_hist = []
    score_filter_hist = []
    if testset.name.startswith("lesion_cv_ap"):
        print("running prediction on valid set...")
    elif testset.name.startswith("lesion_test_ap"):
        print("running prediction on test set...")
    for idx in tqdm(range(num_images)):
        # pull img & annotations
        img = testset.pull_image(idx)
        annotation = [testset.pull_anno(idx)] #min_width, min_height

        # base transform the image and permute to [phase, channel, h, w] and flatten to [1, channel, h, w]
        x = torch.from_numpy(transform(img)[0]).permute(0, 3, 1, 2).contiguous()
        x = x.view(-1, x.shape[2], x.shape[3])
        x = Variable(x).unsqueeze(0)
        if use_pixel_link:
            annotation[0] = annotation[0] * x.size(-1) / img.shape[-2]
        if cuda:
            x = x.cuda()
        if use_pixel_link:
            with torch.no_grad():
                pixel_pos_scores, link_pos_scores = net(x)
                detections = mask_to_box(pixel_pos_scores, link_pos_scores, img_shape=(x.size(-1), x.size(-1)), pixel_thres=thresh)
                if len(detections[0]) == 0:
                    continue
                detections = detections[0]
                score = detections[:, 0]
                coords = detections[:, 1:]
                # TODO: Double-check for detections
                # pixel_pos_scores, link_pos_scores = pixel_pos_scores.cpu().numpy(), link_pos_scores.cpu().numpy()
                # pixel_pos_scores = pixel_pos_scores[:, 1, :, :]
                # link_pos_scores = link_pos_scores[:, 1::2, :, :].transpose((0, 2, 3, 1))
                # mask = decode_batch(pixel_pos_scores, link_pos_scores)[0, ...]
                # bboxes, bboxes_score, pixel_pos_scores = mask_to_bboxes(mask, pixel_pos_scores, (x.shape[2], x.shape[3]))

        else:
            with torch.no_grad():
                if visualize:
                    y, all_offset, all_attnb, all_attn = net(x, visualize=visualize)  # forward pass
                else:
                    y = net(x)

            detections = y.data  # [200, 5]
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

        # filter out boxes with confidence threshold
        score_nofilter_hist.extend(boxes[:, 1])
        boxes = boxes[boxes[:, 1] > thresh]
        score_filter_hist.extend(boxes[:, 1])

        # append ground truth and extend the predictions
        # cut out the label (last elem) since it's not necessary for AP
        predictions.extend(boxes)

        # dump image and offset values
        if visualize:
            assert output_path is not None and model_name is not None
            vis_path = os.path.join(output_path, "visualize", model_name)
            os.makedirs(vis_path, exist_ok=True)
            os.makedirs(os.path.join(vis_path, testset.name), exist_ok=True)
            os.makedirs(os.path.join(vis_path, testset.name+'_jpg'), exist_ok=True)
            x_sav = x.cpu().numpy()
            all_offset = [ofs.cpu().numpy() for ofs in all_offset]
            boxes_ = boxes[:, 1:]
            # to identify which was the gt phase for visualization
            gt_phase = testset.pull_phase(idx)
            img_sav, img_sav_det = prepare_background_img(img, annotation, boxes_, gt_phase)
            all_attn = {str(i): attn.cpu().numpy() for i, attn in enumerate(all_attn)}
            all_attnb = {str(i): attn.cpu().numpy() for i, attn in enumerate(all_attnb)}

            np.save(os.path.join(vis_path, testset.name, str(idx)+'_x.npy'), x_sav)
            np.save(os.path.join(vis_path, testset.name, str(idx) + '_annotation.npy'), annotation)
            np.save(os.path.join(vis_path, testset.name, str(idx) + '_all_offset.npy'), all_offset)
            img_sav = cv2.cvtColor(img_sav.astype(np.uint8), cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(vis_path, testset.name+'_jpg', str(idx) + '_x_cat.jpg'), img_sav)
            img_sav_det = cv2.cvtColor(img_sav_det.astype(np.uint8), cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(vis_path, testset.name+'_jpg', 'det_' +str(idx) + '.jpg'), img_sav_det)
            np.save(os.path.join(vis_path, testset.name, str(idx) + '_all_fusion_attention.npy'), all_attn)
            np.save(os.path.join(vis_path, testset.name, str(idx) + '_all_base_attention.npy'), all_attnb)


        if mode == 'v1':
            # miccai2018 data processing: since then there was 4-phase boxes per data
            # only use the portal phase bbox
            bbox = np.expand_dims(annotation[0][2, :-1], 0)
            det = [False]
            npos += 1
            # bbox = annotation[0][:, :-1]
            # det = [False] * bbox.shape[0]
            # npos += bbox.shape[0]
        elif mode == 'v2':
            # 1904 data processing: now we have single bbox representing for all phases
            # AND, we do have multiple lesions per data
            bbox = annotation[0][:, :-1]
            det = [False] * bbox.shape[0]
            npos += bbox.shape[0]

        class_recs[idx] = {'bbox': bbox,
                           'det': det}

    if testset.name.startswith("lesion_cv_ap"):
        # writer histogram of score map
        if score_nofilter_hist != []:
            writer.add_histogram('score_nofilter', score_nofilter_hist, iteration)
        if score_filter_hist != []:
            writer.add_histogram('score_filter', score_filter_hist, iteration)
        writer.flush()

    # sort the prediction in descending global confidence
    # first parse predictions into img ids, confidence and bb
    image_ids = [x[0] for x in predictions]
    confidence = np.array([float(x[1]) for x in predictions])
    BB = np.array([[float(z) for z in x[2:]] for x in predictions])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    try:
        BB = BB[sorted_ind, :]
    except IndexError:
        print("WARNING: zero bbox found during AP calculation returning 0...")
        return None, None, None, None, None
    image_ids = [int(image_ids[x]) for x in sorted_ind]

    return class_recs, image_ids, confidence, BB, npos





def test_net(net, cuda, testset, transform,
             imsize=300, thresh=0.05, mode='v1', use_07_metric=True, ap_list=[0.5], iobb_list=[0.1],
             writer=None, iteration=None, visualize=False, output_path=None, model_name=None, use_pixel_link=False):
    # we define copy of objects for each metrics
    ap_result, iobb_result = [], []
    class_recs, image_ids, confidence, BB, npos = make_pred(net, cuda, testset, transform, thresh=thresh, mode=mode,
                                                            writer=writer, iteration=iteration,
                                                            visualize=visualize, output_path=output_path, model_name=model_name, use_pixel_link=use_pixel_link)
    if visualize:
        # we're done here
        return None, None

    # edge case handling
    if class_recs is None:
        for i in range(len(ap_list)):
            ap_result.append(0.)
        for i in range(len(iobb_list)):
            iobb_result.append(0.)
        return ap_result, iobb_result

    nd = len(image_ids)
    tp_ap, fp_ap = [np.zeros(nd) for _ in range(len(ap_list))], [np.zeros(nd) for _ in range(len(ap_list))]
    tp_iobb, fp_iobb = [np.zeros(nd) for _ in range(len(iobb_list))], [np.zeros(nd) for _ in range(len(iobb_list))]
    class_rec_ap = [copy.deepcopy(class_recs) for _ in range(len(ap_list))]
    class_rec_iobb = [copy.deepcopy(class_recs) for _ in range(len(iobb_list))]

    print("calculating ap...")
    for d in tqdm(range(nd)):
        # load BBGT from class_recs (NOT from copies of class_rec_ap & iobb): same value but just for avoiding silly mistakes
        BBGT = class_recs[image_ids[d]]['bbox'].astype(float)
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
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
            uni_iou = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                   (BBGT[:, 2] - BBGT[:, 0]) *
                   (BBGT[:, 3] - BBGT[:, 1]) - inters)
            uni_iobb = (bb[2] - bb[0]) * (bb[3] - bb[1])
            overlaps_iou = inters / uni_iou
            overlaps_iobb = inters / uni_iobb
            ovmax_iou, ovmax_iobb = np.max(overlaps_iou), np.max(overlaps_iobb)
            jmax_iou, jmax_iobb = np.argmax(overlaps_iou), np.argmax(overlaps_iobb)

            # update AP list
            for i_ap in range(len(ap_list)):
                R = class_rec_ap[i_ap][image_ids[d]]
                if ovmax_iou > ap_list[i_ap]:
                    if not R['det'][jmax_iou]:
                        tp_ap[i_ap][d] = 1.
                        R['det'][jmax_iou] = 1
                    else:
                        fp_ap[i_ap][d] = 1.
                else:
                    fp_ap[i_ap][d] = 1.
            # update IoBB list
            for i_iobb in range(len(iobb_list)):
                R = class_rec_iobb[i_iobb][image_ids[d]]
                if ovmax_iobb > iobb_list[i_iobb]:
                    if not R['det'][jmax_iobb]:
                        tp_iobb[i_iobb][d] = 1.
                        R['det'][jmax_iobb] = 1
                    else:
                        fp_iobb[i_iobb][d] = 1.
                else:
                    fp_iobb[i_iobb][d] = 1.

    # compute precision recall
    for i_ap in range(len(ap_list)):
        fp_ap[i_ap] = np.cumsum(fp_ap[i_ap])
        tp_ap[i_ap] = np.cumsum(tp_ap[i_ap])

        rec = tp_ap[i_ap] / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp_ap[i_ap] / np.maximum(tp_ap[i_ap] + fp_ap[i_ap], np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric=use_07_metric)
        ap_result.append(ap)

    for i_iobb in range(len(iobb_list)):
        fp_iobb[i_iobb] = np.cumsum(fp_iobb[i_iobb])
        tp_iobb[i_iobb] = np.cumsum(tp_iobb[i_iobb])

        rec = tp_iobb[i_iobb] / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp_iobb[i_iobb] / np.maximum(tp_iobb[i_iobb] + fp_iobb[i_iobb], np.finfo(np.float64).eps)
        ap_iobb = voc_ap(rec, prec, use_07_metric=use_07_metric)
        iobb_result.append(ap_iobb)

    return ap_result, iobb_result
