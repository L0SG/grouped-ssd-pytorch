import numpy as np
import glob
import os
import natsort
import pydicom as dicom
from lib.utils import read_dicom_series
import scipy

"""
# cut out korean characters
for subject in glob.glob(os.path.join(data_path, '*')):
    for subfolder in glob.glob(os.path.join(subject, '*')):
        basename = os.path.basename(subfolder)
        basename_words = basename.split()
        if basename_words[-1] == "폴더":
            basename_new = basename_words[0]
            os.rename(os.path.join(subject, basename),
                      os.path.join(subject, basename_new))
"""


def apply_window(img, window_width, window_level):
    """
    apply CT windowing function in radiology terms
    set upper & lower grey values and clamp the pixels with in the calculated HU bound
    implemented as: https://radiopaedia.org/articles/windowing-ct
    # http://radclass.mudr.org/content/hounsfield-units-scale-hu-ct-numbers
    :param img: raw CT images with shape [4, 512, 512, num_slices] with HU unit
    :param window_width: window width
    :param window_level: window level
    :return:
    """
    if np.amin(img) == 0:
        print("WARNING: This CT image has minimum CT HU of 0. Double check if the windowing function is right")
        print("WARNING: manually adjusting HU to have minimum value of -1024...")
        img = np.subtract(img, 1024)

    img[img > 1200] = 0

    upper_grey = window_level + (window_width / 2.)
    lower_grey = window_level - (window_width / 2.)

    img_windowed = np.clip(img, lower_grey, upper_grey)

    return img_windowed


def read_liver_seg_masks_raw_year2(masks_dirname, img_shape):
    """
    read 3d liver segmentation raw file
    0's for background pixels, 1's for liver pixels
    :param masks_dirname:
    :return:
    """
    label_volume = None
    rawfile = np.fromfile(masks_dirname, dtype='uint8', sep="")

    # raw file assumes height as first dimension
    # permute img_shape and reshape the raw image
    shape_raw = np.array(img_shape)
    order = [2, 0, 1]
    shape_raw = shape_raw[order]
    num_slice = rawfile.shape[0] / shape_raw[1] / shape_raw[2]
    assert float(int(num_slice)) == num_slice
    num_slice = int(num_slice)
    print(os.path.basename(masks_dirname) + ' slices raw vs dicom: ' + str(num_slice) + ' ' + str(shape_raw[0]))

    if num_slice != shape_raw[0]:
        print("WARNING: {} mask slice {} does not match CT image {}".format(os.path.basename(masks_dirname), num_slice,
                                                                            shape_raw[0]))
    # TODO: this is temporary for just seeing the raw dataset
    shape_raw[0] = num_slice

    label_volume = rawfile.reshape(shape_raw)
    label_volume = label_volume.transpose([1, 2, 0])

    # raw seg image is upside-down: do vertical flip
    # label_volume = np.flipud(label_volume)

    # # rotate to the left 3 times: since it seems to be rotated to the right 90
    # label_volume = np.rot90(label_volume, 3)

    return label_volume


def load_liver_seg_dataset_year2(data_path, num_data_to_load, window_width, window_level):
    """
    load the liver dataset
    :param data_path:
    :return: list of 3D CT data (variable height) & list of 3D binary segmentation data
    """

    # get list of subfolder names
    dir_list = os.listdir(data_path)
    dir_list = sorted(dir_list)

    # empty list for appending data for each subject
    list_prect = []
    list_ct = []
    list_mask = []
    count_for_early_stop = 0

    # traverse through subfolders
    # each subfolder contains one raw seg file and two sub-subfolders: CT and pre-CT
    # CT & pre-CT contains A, D, P, Pre, just like the year1 dataset
    list_phase = ['A', 'D', 'P', 'Pre']
    for dir_name in dir_list:
        # obtain absolute path of the subject
        path_subject = os.path.join(data_path, dir_name)
        # load dicom image of the subject
        dicom_image_before = []
        dicom_image_after = []
        for phase in list_phase:
            dicom_image_before.append(read_dicom_series(os.path.join(path_subject, 'pre-CT', phase), phase + "_*"))
            dicom_image_after.append(read_dicom_series(os.path.join(path_subject, 'CT', phase), phase + "_*"))
        # [4, 512, 512, z (variable height)], both before and after must have same z
        dicom_image_before = np.array(dicom_image_before)
        dicom_image_after = np.array(dicom_image_after)
        assert dicom_image_before.shape == dicom_image_after.shape
        print(os.path.basename(path_subject) + " min_prect, max_prect, min_ct, max_ct: " + str(
            np.amin(dicom_image_before)) + ' ' + str(np.amax(dicom_image_before)) + ' ' +
              str(np.amin(dicom_image_after)) + ' ' + str(np.amax(dicom_image_after)))

        # apply HU windowing to the dicom images
        dicom_image_before = apply_window(dicom_image_before, window_width, window_level)
        dicom_image_after = apply_window(dicom_image_after, window_width, window_level)

        # TODO: 1810 dataset has D labels
        # load mask image of the subject, which needs shape information
        dicom_shape = dicom_image_after.shape[1:]
        mask_path = os.path.join(path_subject, str(dir_name) + '.raw')
        # FIXME: 1810 hard-wire
        mask_path = mask_path.replace(".raw", "_D.raw")
        mask_image = read_liver_seg_masks_raw_year2(mask_path, img_shape=dicom_shape)

        # print unique elements of mask tensor
        # for binary segmentation, it should only contain 0 or 1
        # if the tensor contains other number, data structure is not right
        mask_unique_elems = np.unique(mask_image)
        ground_truth_reference = [0, 1]  # to assert binary data
        # print('unique elements of mask image: ' + str(mask_unique_elems))
        if sorted(mask_unique_elems) != sorted(ground_truth_reference):
            print('incorrect mask label detected, forcing incorrect labels to zero...')
            # currently hard-coding (10 and 20)
            # TODO: make this generic
            mask_image[mask_image == 10] = 0
            mask_image[mask_image == 20] = 0
        # check label sanity again and assert
        mask_unique_elems = np.unique(mask_image)
        assert len(mask_unique_elems) == len(ground_truth_reference)
        assert sorted(mask_unique_elems) == sorted(ground_truth_reference)

        # append to list
        list_prect.append(dicom_image_before)
        list_ct.append(dicom_image_after)
        # list_mask.append(mask_image)

        # DEBUG ONLY: save dicom & mask images for given directory
        # for checking if the label data has any artifacts
        # save mask data as image
        mask_image_save_path = '/home/tkdrlf9202/Datasets/liver_mask_image_year2_1810_ww{}_wl{}'. \
            format(window_width, window_level)
        if not os.path.exists(os.path.join(mask_image_save_path, str(dir_name))):
            os.makedirs(os.path.join(mask_image_save_path, str(dir_name)))

        # take the maximum one: for outputting concatenated image
        z_max = np.max([dicom_image_before.shape[3], dicom_image_after.shape[3], mask_image.shape[2]])

        for idx in range(z_max):
            cat_slice_save_path = os.path.join(mask_image_save_path, str(dir_name), str(idx) + '.jpg')
            cat_slice = np.zeros([512 * 2, 512 * 5])


            try:
                dicom_slice_before = np.copy(dicom_image_before[:, :, :, idx])
                dicom_slice_after = np.copy(dicom_image_after[:, :, :, idx])
            except IndexError:
                # ct image not found (since ct vs mask is not aligned), replace with noise
                dicom_slice_before = np.random.randint(0, 255, size=(4, 512, 512))
                dicom_slice_after = np.random.randint(0, 255, size=(4, 512, 512))

            dicom_slice_before_normalized = (dicom_slice_before - float(np.amin(dicom_slice_before))) / (
                    float(np.amax(dicom_slice_before)) - float(np.amin(dicom_slice_before)))
            dicom_slice_after_normalized = (dicom_slice_after - float(np.amin(dicom_slice_after))) / (
                    float(np.amax(dicom_slice_after)) - float(np.amin(dicom_slice_after)))
            dicom_slice_before_normalized *= 255
            dicom_slice_after_normalized *= 255

            try:
                mask_slice = np.copy(mask_image[:, :, idx])
                # if 0, stay 0; if 1, convert to 255
                mask_slice[mask_slice == 1] = 255
            except IndexError:
                # mask slice not found (ct vs mask not aligned), replace with noise
                mask_slice = np.random.randint(0, 255, size=(512, 512))

            """
            cat_slice[:, 0:512] = dicom_slice_before_normalized
            cat_slice[:, 512:512*2] = dicom_slice_after_normalized
            cat_slice[:, 512*2:] = mask_slice
            """
            # for displaying all 4 phases
            dicom_slice_before_normalized = np.split(dicom_slice_before_normalized, 4, axis=0)
            dicom_slice_after_normalized = np.split(dicom_slice_after_normalized, 4, axis=0)
            for i in range(4):
                cat_slice[:512, 512 * (i):512 * (i + 1)] = dicom_slice_before_normalized[i]
                cat_slice[512:, 512 * (i):512 * (i + 1)] = dicom_slice_after_normalized[i]
            # display mask
            cat_slice[512:, 512 * 4:] = mask_slice
            scipy.misc.imsave(cat_slice_save_path, cat_slice)

        # # traverse through slices
        # for idx in range(mask_image.shape[2]):
        #     mask_slice_save_path = os.path.join(mask_image_save_path, str(dir_name), 'mask_'+str(idx)+'.jpg')
        #     mask_slice = np.copy(mask_image[:,:,idx])
        #     # if 0, stay 0; if 1, convert to 255, if other, convert to 128(grey)
        #     mask_slice[mask_slice == 1] = 255
        #     # mask_slice[mask_slice == 10] = 100
        #     # mask_slice[mask_slice == 20] = 180
        #     scipy.misc.imsave(mask_slice_save_path, mask_slice)
        #
        # # save portal dicom image as jpg
        # for idx in range(dicom_image_before.shape[3]):
        #     dicom_slice_save_path = os.path.join(mask_image_save_path, str(dir_name), 'dicom_prect_'+str(idx)+'.jpg')
        #     dicom_slice = np.copy(dicom_image_before[2, :, :, idx])
        #     dicom_slice_normalized = (dicom_slice - float(np.amin(dicom_slice))) / (float(np.amax(dicom_slice)) - float(np.amin(dicom_slice)))
        #     dicom_slice_normalized *= 255
        #     scipy.misc.imsave(dicom_slice_save_path, dicom_slice)
        #
        #     dicom_slice_save_path = os.path.join(mask_image_save_path, str(dir_name), 'dicom_ct_'+str(idx)+'.jpg')
        #     dicom_slice = np.copy(dicom_image_after[2, :, :, idx])
        #     dicom_slice_normalized = (dicom_slice - float(np.amin(dicom_slice))) / (float(np.amax(dicom_slice)) - float(np.amin(dicom_slice)))
        #     dicom_slice_normalized *= 255
        #     scipy.misc.imsave(dicom_slice_save_path, dicom_slice)

        # partial load defined in num_data_to_load
        count_for_early_stop += 1
        if num_data_to_load is None:
            continue
        if count_for_early_stop == num_data_to_load:
            break

    return list_prect, list_ct, list_mask


data_path = "/home/tkdrlf9202/Datasets/liver_year2_dataset_1810"
prect, ct, mask = load_liver_seg_dataset_year2(data_path, None, window_width=400, window_level=50)

print("done")
