# NEW impl for 1904 dataset: things are getting too big, use online data loading from disk!

# for arbitrary image dataset
# courtesy of https://github.com/amdegroot/ssd.pytorch/issues/72

# # for debugging perf hotspot
# from line_profiler import LineProfiler
# from utils import augmentations
import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
# import cv2
import numpy as np

# if sys.version_info[0] == 2:
#    import xml.etree.cElementTree as ET
# else:
#    import xml.etree.ElementTree as ET


LABELS = ['lesion']

LABELS_2_IND = {
    'lesion': 0
}

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))




class DataSplitter():
    def __init__(self, data_path, cross_validation=5, num_test_subject=10):

        # self.root = root
        self.data_path = data_path
        self.metadata_path = os.path.join(self.data_path, "metadata.txt")
        self.data = []
        self.subjects = []
        with open(self.metadata_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split("|")
                self.data.append(line)
                # append subject only
                self.subjects.append(line[1])
        self.subjects = sorted(list(set(self.subjects)))

        self.cross_validation = cross_validation
        self.num_test_subject = num_test_subject
        self.subjects_train = self.subjects[:-num_test_subject]
        self.subjects_test = self.subjects[-num_test_subject:]

        self.subjects_cv_eval = []
        self.subjects_cv_train = []
        if cross_validation != 1:
            divider = int(len(self.subjects_train) / cross_validation)
        elif cross_validation == 1:
            print("INFO: cross_validation set to 0. splitting data into single train-val set... (val defaults to 0.2)")
            divider = int(len(self.subjects_train) * 0.2)

        for i_cv in range(self.cross_validation):
            self.subjects_cv_eval.append(self.subjects_train[divider * i_cv: min(divider * (i_cv+1), len(self.subjects_train))])
            self.subjects_cv_train.append(list(filter(lambda x: x not in self.subjects_cv_eval[i_cv], self.subjects_train)))

        self.data_train = list(filter(lambda x: x[1] in self.subjects_train, self.data))
        self.data_test = list(filter(lambda x: x[1] in self.subjects_test, self.data))

        self.data_cv_eval = []
        self.data_cv_train = []
        for i_cv in range(self.cross_validation):
            self.data_cv_eval.append(list(filter(lambda x: x[1] in self.subjects_cv_eval[i_cv], self.data_train)))
            self.data_cv_train.append(list(filter(lambda x: x[1] in self.subjects_cv_train[i_cv], self.data_train)))

        # self.data_mean = self.get_data_mean()
        # print("calculated data_mean: {}".format(self.data_mean))

    def get_data_mean(self):
        print("calculating data mean...")
        # calcaulte mean pixel value of data by iterating through all "training" dataset
        mean_pixel = []
        for index in range(len(self.data_train)):
            img = np.load(os.path.join(self.data_path, self.data_train[index][0] + "_ct.npy")) * 255.
            mean = img.mean()
            mean_pixel.append(mean)
        mean_pixel = np.array(mean_pixel).mean()
        return mean_pixel



class FISHdetectionV2(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, data_path, data, transform=None, dataset_name='fish_detection', load_data_to_ram=False, debug=False, use_pixel_link=False):

        # self.data_path will recieve root absolute path of the dataset
        self.data_path = data_path
        # self.data will receive partitioned list of paths from DataSplitter
        self.data = [x[0] for x in data]

        self.load_data_to_ram = load_data_to_ram
        if self.load_data_to_ram:
            print("loading data to ram...")
            self.data_img = []
            self.data_target = []
            self.data_phase = []
            for i in range(len(data)):
                img = np.load(os.path.join(self.data_path, self.data[i] + "_ct.npy"))
                # convert to [phase, H, W, C]
                img = np.transpose(img, [0, 2, 3, 1])
                # make it 0~255 uint8
                img = (img * 255).astype(np.uint8)

                # target = ET.parse(self._annopath % img_id).getroot()
                target = np.load(os.path.join(self.data_path, self.data[i] + "_bbox.npy"))
                # cast to float32 for proper scaling
                target = target.astype(np.float32)

                phase = np.load(os.path.join(self.data_path, self.data[i] + "_phase.npy"))

                self.data_img.append(img)
                self.data_target.append(target)
                self.data_phase.append(phase)

                if debug:
                    import cv2
                    import torchvision.utils as vutils

                    img_middle = img[:, :, :, 1]
                    # consider single image as [N, 1, H, W]
                    image_for_vis = torch.tensor(img_middle).unsqueeze(1)
                    img_out = []
                    target_for_vis = torch.tensor(target)
                    for j in range(image_for_vis.shape[0]):
                        img_rgb = np.repeat(image_for_vis[j, :, :], 3, axis=0).permute(1, 2, 0).numpy().copy()
                        for k in range(target_for_vis.shape[0]):
                            xmin, ymin, xmax, ymax = (target_for_vis[k][:4]).long().numpy()
                            cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
                        img_out.append(torch.tensor(img_rgb))
                    img_out = torch.stack(img_out).permute(0, 3, 1, 2)

                    input_visual = vutils.make_grid(img_out.float(), normalize=True, scale_each=True)
                    print("test")




        self.transform = transform
        # self.target_transform = target_transform

        self.name = dataset_name
        # self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        # self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')

        # self.ids = list()
        # for (year, name) in image_sets:
        #    rootpath = os.path.join(self.root, 'VOC' + year)
        #    for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        #        self.ids.append((rootpath, line.strip()))

        self.use_pixel_link = use_pixel_link
        if self.use_pixel_link:
            print("INFO: using pixel_link version of dataset!")

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.data)

    def pull_item(self, index):
        # TODO: double-check new implementations are corrent
        # img_id = self.ids[index]

        if self.load_data_to_ram:
            img = self.data_img[index]
            target = self.data_target[index]
        else:
            #img_path = self.image_paths[index]
            img = np.load(os.path.join(self.data_path, self.data[index] + "_ct.npy"))
            # convert to [phase, H, W, C]
            img = np.transpose(img, [0, 2, 3, 1])
            # make it 0~255 uint8
            img = (img * 255).astype(np.uint8)

            # target = ET.parse(self._annopath % img_id).getroot()
            target = np.load(os.path.join(self.data_path, self.data[index] + "_bbox.npy"))
            # cast to float32 for proper scaling
            target = target.astype(np.float32)
            # target = np.asarray(target).reshape(-1, 5)

        try:
            #img = cv2.imread(img_path)
            # input img_path is already numpy
            # this is unnecessary overhead
            # img = img_path

            # single phase image
            if len(img.shape) == 3:
                height, width, channels = img.shape
            # multi-phase image: discard phase info
            elif len(img.shape) == 4:
                _, height, width, channels = img.shape
        except:
            raise NotImplementedError
            # img = cv2.imread('random_image.jpg')
            # height, width, channels = img.shape

            # if self.target_transform is not None:
        #    target = self.target_transform(target, width, height)


        if self.transform is not None:
            #target = np.array(target)
            """
            # multi-phase data have 4 ground truth
            # randomly select one target
            # TODO: fix this if the data is multi-class target since this assumes that all labels are the same
            rng = np.random.randint(low=0, high=target.shape[0]+1)
            target = target[rng]
            """
            """
            # TODO: fix this later, generic dataset have multiple lesions
            # currently data have only 1 lesion, unsqueeze first dim
            target = np.expand_dims(target, 0)
            """
            # scale each coord from absolute pixels to 0~1
            for idx in range(target.shape[0]):
                """
                x_center, y_center, h, w, cls = target[idx]
                x_center /= height
                y_center /= width
                h /= height
                w /= width
                # WARNING: SSDAugmentation uses y_center as first elem, change it properly
                target[idx] = np.array([y_center, x_center, w, h, cls])
                """
                x_min, y_min, x_max, y_max, cls = target[idx]
                x_min, x_max = x_min/width, x_max/width
                y_min, y_max = y_min/height, y_max/height
                target[idx] = np.array([x_min, y_min, x_max, y_max, cls])

            # # debug hotspot
            # lp = LineProfiler()
            # lp.add_function(augmentations.ConvertFromInts.__call__)
            # lp.add_function(augmentations.ToAbsoluteCoords.__call__)
            # lp.add_function(augmentations.PixelJitter.__call__)
            # lp.add_function(augmentations.PhotometricDistort.__call__)
            # lp.add_function(augmentations.Expand.__call__)
            # lp.add_function(augmentations.RandomSampleCrop.__call__)
            # lp.add_function(augmentations.RandomMirror.__call__)
            # lp.add_function(augmentations.ToPercentCoords.__call__)
            # lp.add_function(augmentations.Resize.__call__)
            # lp.add_function(augmentations.SubtractMeans.__call__)
            # lp_wrapper = lp(self.transform.__call__)
            # lp_wrapper(img, target[:, :4], target[:, 4])
            # lp.print_stats()
            # exit()

            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            """ our data is not rgb
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            """
            if self.use_pixel_link:
                target = np.hstack((boxes, np.expand_dims(labels["labels"], axis=1)))
                labels["boxes"] = target
                target = labels
            else:
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # single-phase case
        if len(img.shape) == 3:
            return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # multi-phase case
        if len(img.shape) == 4:
            img_torch = torch.from_numpy(img).permute(0, 3, 1, 2)

            """ contiguous call for CPU seems slow, collapse after CUDA instead
            img_torch = torch.from_numpy(img).permute(0, 3, 1, 2).contiguous()
            # collapse phase & channel
            img_torch = img_torch.view(-1, img_torch.shape[2], img_torch.shape[3])
            """
            return img_torch, target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        # img_id = self.ids[index]
        if self.load_data_to_ram:
            img = self.data_img[index]
        else:
            img = np.load(os.path.join(self.data_path, self.data[index] + "_ct.npy"))
            # convert to [phase, H, W, C]
            img = np.transpose(img, [0, 2, 3, 1])
            # make it 0~255 uint8
            img = (img * 255).astype(np.uint8)

        # ct data is already numpy
        #return cv2.imread(img_path, cv2.IMREAD_COLOR)
        return img

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        # img_id = self.ids[index]
        # anno = ET.parse(self._annopath % img_id).getroot()
        # gt = self.target_transform(anno, 1, 1)
        #img_path = self.image_paths[index]
        # target = ET.parse(self._annopath % img_id).getroot()
        if self.load_data_to_ram:
            target = self.data_target[index]
        else:
            target = np.load(os.path.join(self.data_path, self.data[index] + "_bbox.npy"))
            # cast to float32 for proper scaling
            target = target.astype(np.float32)

        #return img_path, target
        return target

    def pull_phase(self, index):
        phase = np.load(os.path.join(self.data_path, self.data[index] + "_phase.npy"))
        return phase

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


def detection_collate_v2(batch, p_only=False):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def detection_collate_v2_pixel_link(batch, p_only=False):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    pixel_mask, neg_pixel_mask, labels, pixel_pos_weight, link_mask, boxes = [], [], [], [], [], []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        sample_1 = sample[1]
        pixel_mask.append(torch.LongTensor(sample_1["pixel_mask"]))
        neg_pixel_mask.append(torch.LongTensor(sample_1["neg_pixel_mask"]))
        labels.append(torch.FloatTensor(sample_1["labels"]))
        pixel_pos_weight.append(torch.FloatTensor(sample_1["pixel_pos_weight"]))
        link_mask.append(torch.LongTensor(sample_1["link_mask"]))
        boxes.append(torch.FloatTensor(sample_1["boxes"]))

    imgs = torch.stack(imgs, 0)
    pixel_mask = torch.stack(pixel_mask, 0)
    neg_pixel_mask = torch.stack(neg_pixel_mask, 0)
    pixel_pos_weight = torch.stack(pixel_pos_weight, 0)
    link_mask = torch.stack(link_mask, 0)

    # no stacking for labels and boxes: variable number of lesions

    targets = {'pixel_mask': pixel_mask, 'neg_pixel_mask': neg_pixel_mask,
                'pixel_pos_weight': pixel_pos_weight, 'link_mask': link_mask, 'lables': labels, 'boxes': boxes}
    return imgs, targets




if __name__ == "__main__":
    from utils.augmentations import SSDAugmentation
    data_path = "/home/tkdrlf9202/Datasets/liver_year1_dataset_extended_1904_preprocessed/ml_ready"
    debug_dataset = DataSplitter(data_path, cross_validation=1, num_test_subject=10)
    debug_dataset_cv0_train = debug_dataset.data_cv_train[0]
    debug_dataset_cv0_eval = debug_dataset.data_cv_eval[0]

    debug_fishdataset = FISHdetectionV2(data_path, debug_dataset_cv0_train, SSDAugmentation(), load_data_to_ram=True, debug=True)
    test_batch = next(iter(debug_fishdataset))