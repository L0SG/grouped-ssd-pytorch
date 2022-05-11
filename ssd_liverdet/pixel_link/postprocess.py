import torch
import cv2
import numpy as np
import torch.nn as nn
import pixel_link.pixel_link_config as config

def mask_filter(pixel_mask, link_mask, neighbors=8, scale=4):
    """
    pixel_mask: batch_size * 2 * H * W
    link_mask: batch_size * 16 * H * W
    """
    batch_size = link_mask.size(0)
    mask_height = link_mask.size(2)
    mask_width = link_mask.size(3)
    pixel_class = nn.Softmax2d()(pixel_mask)
    # print(pixel_class.shape)
    pixel_class = pixel_class[:, 1] > 0.7
    # print(pixel_class.shape)
    # pixel_class = pixel_mask[:, 1] > pixel_mask[:, 0]
    # link_neighbors = torch.ByteTensor([batch_size, neighbors, mask_height, mask_width])
    link_neighbors = torch.zeros([batch_size, neighbors, mask_height, mask_width], \
                                    dtype=torch.uint8, device=pixel_mask.device)
    
    for i in range(neighbors):
        # print(link_mask[:, [2 * i, 2 * i + 1]].shape)
        tmp = nn.Softmax2d()(link_mask[:, [2 * i, 2 * i + 1]])
        # print(tmp.shape)
        link_neighbors[:, i] = tmp[:, 1] > 0.7
        # link_neighbors[:, i] = link_mask[:, 2 * i + 1] > link_mask[:, 2 * i] 
        link_neighbors[:, i] = link_neighbors[:, i] & pixel_class
    # res_mask = np.zeros([batch_size, mask_height, mask_width], dtype=np.uint8)
    pixel_class = pixel_class.cpu().numpy()
    link_neighbors = link_neighbors.cpu().numpy()
    return pixel_class, link_neighbors

def min_area_rect(cnt):
    """
    Args:
        xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes. 4 contains [x1, x2, x3, x4]
        ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
            Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an oriented bbox.
    Return:
        the oriented rects sorrounding the box, in the format:[cx, cy, w, h, theta].
    """
    rect = cv2.minAreaRect(cnt)
    cx, cy = rect[0]
    w, h = rect[1]
    theta = rect[2]
    box = [cx, cy, w, h, theta]
    return box, w * h

def rect_to_xys(rect, image_shape):
    """Convert rect to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid, i.e., within image area
    """
    h, w = image_shape[0:2]

    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x

    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y

    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    for i_xy, (x, y) in enumerate(points):
        x = get_valid_x(x)
        y = get_valid_y(y)
        points[i_xy, :] = [x, y]
    points = np.reshape(points, -1)
    return points

def mask_to_box(pixel_mask, link_mask, neighbors=8, img_shape=(300,300), pixel_thres=None):
    """
    pixel_mask: batch_size * 2 * H * W
    link_mask: batch_size * 16 * H * W
    """
    def distance(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    def short_side_filter(bounding_box):
        for i, point in enumerate(bounding_box):
            if distance(point, bounding_box[(i+1)%4]) < 5**2:
                return True # ignore it
        return False # do not ignore
    batch_size = link_mask.size(0)
    mask_height = link_mask.size(2)
    mask_width = link_mask.size(3)
    pixel_class = nn.Softmax2d()(pixel_mask)
    score_maps = pixel_class[:, 1]
    # print(pixel_class.shape)
    if pixel_thres is None:
        pixel_class = pixel_class[:, 1] > config.pixel_conf_threshold
    else:
        pixel_class = pixel_class[:, 1] > pixel_thres
    # pixel_class = pixel_mask[:, 1] > pixel_mask[:, 0]
    # link_neighbors = torch.ByteTensor([batch_size, neighbors, mask_height, mask_width])
    link_neighbors = torch.zeros([batch_size, neighbors, mask_height, mask_width], \
                                    dtype=torch.uint8, device=pixel_mask.device)
    
    for i in range(neighbors):
        # print(link_mask[:, [2 * i, 2 * i + 1]].shape)
        tmp = nn.Softmax2d()(link_mask[:, [2 * i, 2 * i + 1]])
        # print(tmp.shape)
        link_neighbors[:, i] = tmp[:, 1] > config.link_conf_threshold
        # link_neighbors[:, i] = link_mask[:, 2 * i + 1] > link_mask[:, 2 * i] 
        link_neighbors[:, i] = link_neighbors[:, i] & pixel_class.byte()
    # res_mask = np.zeros([batch_size, mask_height, mask_width], dtype=np.uint8)
    all_boxes = []
    all_scores = []
    # res_masks = []
    for i in range(batch_size):
        res_mask = func(pixel_class[i], link_neighbors[i])

        res_mask = cv2.resize(res_mask, img_shape, interpolation=cv2.INTER_NEAREST)
        score_map = cv2.resize(score_maps[i].detach().cpu().numpy(), img_shape, interpolation=cv2.INTER_LINEAR)

        box_num = np.amax(res_mask)
        # print(res_mask.any())
        bounding_boxes = []
        scores = []
        for i in range(1, box_num + 1):
            box_mask_np = (res_mask == i).astype(np.uint8)
            # res_masks.append(box_mask)
            if box_mask_np.sum() < 100:
                pass
                # print("<150")
                # continue
            try:
                contours, _ = cv2.findContours(box_mask_np, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
            except:
                _, contours, _ = cv2.findContours(box_mask_np, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
            # print(contours[0])
            rect, rect_area = min_area_rect(contours[0])

            w, h = rect[2:-1]
            if min(w, h) < config.min_height:
                continue
            if rect_area < config.min_area:
                continue
            bbox = rect_to_xys(rect, img_shape)
            min_x = np.min([bbox[0], bbox[2], bbox[4], bbox[6]])
            max_x = np.max([bbox[0], bbox[2], bbox[4], bbox[6]])
            min_y = np.min([bbox[1], bbox[3], bbox[5], bbox[7]])
            max_y = np.max([bbox[1], bbox[3], bbox[5], bbox[7]])
            bounding_boxes.append([min_x, min_y, max_x, max_y])
            xs, ys = np.where(box_mask_np)
            scores.append([np.mean(score_map[xs, ys])])
            # if short_side_filter(bounding_box):
                # print("<5")
                # pass
                # continue
            # bounding_box = bounding_box.reshape(8)
            # bounding_box = np.clip(bounding_box * scale, 0, 128 * scale - 1).astype(np.int)
            # import IPython
            # IPython.embed()
            # bounding_boxes.append(bounding_box)
        all_boxes.append(bounding_boxes)
        all_scores.append(scores)

        all_detections = np.concatenate((np.array(all_scores), np.array(all_boxes)), axis=-1)
    return all_detections

def get_neighbors(h_index, w_index):
    res = []
    res.append((h_index - 1, w_index - 1))
    res.append((h_index - 1, w_index))
    res.append((h_index - 1, w_index + 1))
    res.append((h_index, w_index + 1))
    res.append((h_index + 1, w_index + 1))
    res.append((h_index + 1, w_index))
    res.append((h_index + 1, w_index - 1))
    res.append((h_index, w_index - 1))
    return res

def func(pixel_cls, link_cls):
    def joint(pointa, pointb):
        roota = find_root(pointa)
        rootb = find_root(pointb)
        if roota != rootb:
            group_mask[rootb] = roota
            # group_mask[pointb] = roota
            # group_mask[pointa] = roota
        return

    def find_root(pointa):
        root = pointa
        while group_mask.get(root) != -1:
            root = group_mask.get(root)
        return root

    pixel_cls = pixel_cls.cpu().numpy()
    link_cls = link_cls.cpu().numpy()

    # import IPython
    # IPython.embed()

    # print(pixel_cls.any())
    # print(np.where(pixel_cls))
    pixel_points = list(zip(*np.where(pixel_cls)))
    h, w = pixel_cls.shape
    group_mask = dict.fromkeys(pixel_points, -1)
    # print(group_mask)

    for point in pixel_points:
        h_index, w_index = point
        # print(point)
        neighbors = get_neighbors(h_index, w_index)
        for i, neighbor in enumerate(neighbors):
            nh_index, nw_index = neighbor
            if nh_index < 0 or nw_index < 0 or nh_index >= h or nw_index >= w:
                continue
            if pixel_cls[nh_index, nw_index] == 1 and link_cls[i, h_index, w_index] == 1:
                joint(point, neighbor)

    res = np.zeros(pixel_cls.shape, dtype=np.uint8)
    root_map = {}
    for point in pixel_points:
        h_index, w_index = point
        root = find_root(point)
        if root not in root_map:
            root_map[root] = len(root_map) + 1
        res[h_index, w_index] = root_map[root]

    return res

