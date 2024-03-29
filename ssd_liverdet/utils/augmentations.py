import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from pixel_link.pixellink_data import *
from PIL import Image

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        if len(image.shape) == 3:
            height, width, channels = image.shape
        elif len(image.shape) == 4:
            _, height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class PixelJitter(object):
    def __init__(self, percentage):
        self.percentage = percentage

    def __call__(self, image, boxes=None, labels=None):
        # add pixel jitter for targets for data augmentation
        # it applies random pixel shift (up to 1%) for each element of [x_min, y_min, x_max, y_max] & keep label info
        if len(image.shape) == 3:
            height, width, channels = image.shape
        elif len(image.shape) == 4:
            _, height, width, channels = image.shape
        label_noise = np.random.uniform(-self.percentage, self.percentage, size=boxes.shape)
        label_noise[:, 0] *= width
        label_noise[:, 1] *= height
        label_noise[:, 2] *= width
        label_noise[:, 3] *= height
        label_noise = label_noise.astype(np.int8).astype(np.float32)
        boxes_augment = np.add(boxes, label_noise)
        try:
            assert np.all(boxes_augment[:, 0] < boxes_augment[:, 2]) and np.all(boxes_augment[:, 1] < boxes_augment[:, 3])
        except AssertionError:
            # fallback to original boxes to prevent "too dirty" labels which causes NaN loss
            boxes_augment = boxes

        return image, boxes_augment, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        if len(image.shape) == 3:
            height, width, channels = image.shape
        elif len(image.shape) == 4:
            _, height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def _resize_image(image, target):
        return cv2.resize(image, dsize=(target[0], target[1]))

    def __call__(self, image, boxes=None, labels=None):

        if len(image.shape) == 3:
            image = cv2.resize(image, (self.size,
                                     self.size))
            return image, boxes, labels

        elif len(image.shape) == 4:
            image_resized = np.zeros([image.shape[0], self.size, self.size, image.shape[3]], dtype=image.dtype)
            for idx in range(image.shape[0]):
                image_resized[idx] = cv2.resize(image[idx], (self.size, self.size))
            return image_resized, boxes, labels


class ResizeFast(object):
    def __init__(self, size=300):
        self.size = size

    def _resize_image(image, target):
        return cv2.resize(image, dsize=(target[0], target[1]))

    def __call__(self, image, boxes=None, labels=None):

        if len(image.shape) == 3:
            image = cv2.resize(image, (self.size,
                                     self.size))
            return image, boxes, labels

        elif len(image.shape) == 4:
            image_resized = np.zeros([image.shape[0], self.size, self.size, image.shape[3]], dtype=image.dtype)
            for idx in range(image.shape[0]):
                img = (image[idx] * 255).astype(np.uint8)
                img = Image.fromarray(img).resize((self.size, self.size))
                img = np.asarray(img).astype(np.float32) / 255.
                image_resized[idx] = img
                del img
            return image_resized, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels

"""
class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels
"""

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        if len(image.shape) == 3:
            height, width, _ = image.shape
        elif len(image.shape) == 4:
            _, height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(0, width - w)
                top = random.uniform(0, height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                if len(image.shape) == 3:
                    current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                elif len(image.shape) == 4:
                    current_image = current_image[:, rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean, ratio):
        self.mean = mean
        self.ratio = ratio

    def __call__(self, image, boxes, labels):
        #if random.randint(0, 2):
        #    return image, boxes, labels

        # single-phase case: original
        if len(image.shape) == 3:
            height, width, depth = image.shape
            ratio = random.uniform(1, self.ratio)
            left = random.uniform(0, width*ratio - width)
            top = random.uniform(0, height*ratio - height)

            expand_image = np.zeros(
                (int(height*ratio), int(width*ratio), depth),
                dtype=image.dtype)
            expand_image[:, :, :] = self.mean
            expand_image[int(top):int(top + height),
                         int(left):int(left + width)] = image
            image = expand_image

            boxes = boxes.copy()
            boxes[:, :2] += (int(left), int(top))
            boxes[:, 2:] += (int(left), int(top))

            return image, boxes, labels

        # multi-channel case: same expansion scheme for each phase
        elif len(image.shape) == 4:
            phase, height, width, depth = image.shape
            ratio = random.uniform(1, self.ratio)
            left = random.uniform(0, width * ratio - width)
            top = random.uniform(0, height * ratio - height)

            expand_image = np.zeros(
                (phase, int(height * ratio), int(width * ratio), depth),
                dtype=image.dtype)
            expand_image[:, :, :, :] = self.mean
            expand_image[:, int(top):int(top + height),
                         int(left):int(left + width), :] = image
            image = expand_image
            boxes = boxes.copy()
            boxes[:, :2] += (int(left), int(top))
            boxes[:, 2:] += (int(left), int(top))
            return image, boxes, labels

class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        if len(image.shape) == 3:
            _, width, _ = image.shape
            if random.randint(0, 2):
                image = image[:, ::-1]
                boxes = boxes.copy()
                boxes[:, 0::2] = width - boxes[:, 2::-2]
            return image, boxes, classes

        elif len(image.shape) == 4:
            _, _, width, _ = image.shape
            if random.randint(0, 2):
                image = image[:, :, ::-1]
                boxes = boxes.copy()
                boxes[:, 0::2] = width - boxes[:, 2::-2]
            return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            # CT is not RGB: disable color convert, saturation & hue randomization
            # ConvertColor(transform='HSV'),
            # RandomSaturation(),
            # RandomHue(),
            # ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()

        # each channels should not be shuffled for CT
        #self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        # since convertcolor & rs & rh are disabled, both distort are same: rc only
        if random.randint(0, 2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        # do not use lightning noise: CT channels are z-axis
        # return self.rand_light_noise(im, boxes, labels)
        return im, boxes, labels


class POnly(object):
    def __call__(self, image, boxes=None, labels=None):
        # drop other phases other than portal (index 2)
        image = np.repeat(np.expand_dims(image[2], 0), 4, axis=0)
        return image, boxes, labels


class Normalize(object):
    def __call__(self, image, boxes=None, labels=None):
        img_min = image.min()
        img_max = image.max()
        assert img_min != img_max, "all-black image detected during Normalizing. check preprocessing"
        img_norm = (image - img_min) / (img_max - img_min)
        return img_norm, boxes, labels


class PreparePixelLinkTargets(object):
    def __init__(self, size, pixel_link_version="2s"):
        self.size = size
        self.pixel_link_version = pixel_link_version

    def __call__(self, image, boxes=None, labels=None):
        # given SSD targets annotation, generate labels used for pixellink
        # input: targets list with shape [batch_size]
        # each element is tensor that has [x_min, y_min, x_max, y_max, label (0.0)]] in percentage coordinate
        # convert to [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max] in int list

        boxes_long = np.array(boxes * self.size, dtype=np.long)
        boxes_converted = np.take(boxes_long, indices=[0, 1, 2, 1, 2, 3, 0, 3], axis=1)
        pixel_mask, neg_pixel_mask, pixel_pos_weight, link_mask = label_to_mask_and_pixel_pos_weight(boxes_converted, (self.size, self.size),
                                                                                                     self.pixel_link_version)

        labels = {'pixel_mask': pixel_mask, 'neg_pixel_mask': neg_pixel_mask, 'labels': labels,
                           'pixel_pos_weight': pixel_pos_weight, 'link_mask': link_mask}

        return image, boxes, labels

class SSDAugmentation(object):
    def __init__(self, pixeljitter=0.01, ratio=1.5, size=300, mean=(104, 117, 123), use_normalize=False, p_only=False, use_pixel_link=False, pixel_link_version="2s"):
        self.pixeljitter = pixeljitter
        self.mean = mean
        self.size = size
        self.ratio = ratio
        self.use_normalize = use_normalize
        self.p_only = p_only
        self.use_pixel_link = use_pixel_link
        compose_list = [
            ConvertFromInts(),
            ToAbsoluteCoords(),
            # new augmentation: pixel jitter
            PixelJitter(self.pixeljitter),
            PhotometricDistort(),
            Expand(self.mean, self.ratio),
            # TODO: consider these augmentations for CT
            # cropping seems to be not good for CT
            RandomSampleCrop(),
            # mirroring is not correct method for CT
            RandomMirror(),
            ToPercentCoords(),
            SubtractMeans(self.mean)
        ]
        if self.p_only:
            compose_list.append(POnly())
        if self.use_normalize:
            compose_list.append(Normalize())

        # compose_list.append(Resize(self.size))

        assert self.use_normalize, "new ResizeFast implementation assumes --use_normalize to True!"
        compose_list.append(ResizeFast(self.size))

        if self.use_pixel_link:
            print("INFO:using augmentation modified for pixellink model!")
            self.pixel_link_version = pixel_link_version
            compose_list.append(PreparePixelLinkTargets(self.size, self.pixel_link_version))
        self.augment = Compose(compose_list)

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)

