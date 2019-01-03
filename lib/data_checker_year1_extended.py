# sanity checker script for year1 extended dataset received in 1809
import numpy as np
import glob
import os
import natsort
import pydicom as dicom
import scipy
from scipy.misc import imsave

def read_dicom_series(directory, filepattern="P_*"):
    """ Reads a DICOM Series files in the given directory.
    Only filesnames matching filepattern will be considered"""

    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise ValueError("Given directory does not exist or is a file : " + str(directory))
    # print('\tRead Dicom', directory)
    lstFilesDCM = natsort.natsorted(glob.glob(os.path.join(directory, filepattern)))
    # print('\tLength dicom series', len(lstFilesDCM))
    # Get ref file
    RefDs = dicom.read_file(lstFilesDCM[0])
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

    return ArrayDicom


def read_liver_seg_masks_raw_year1_extended(masks_dirname):
    """
    read 3d liver segmentation raw file
    0's for background pixels, 1's for liver pixels
    :param masks_dirname:
    :return:
    """
    label_volume = None
    rawfile = np.fromfile(masks_dirname, dtype='uint8', sep="")
    # check if this rawfile is binary
    assert np.array_equal(rawfile, rawfile.astype(bool))

    # raw file assumes height as first dimension
    # width & height is 512 x 512, so divide the size of raw data by 512^2, then we get the z size
    z_size = int(rawfile.size / 512 / 512)

    # reshape the raw image and permute
    shape_raw = np.array([z_size, 512, 512])
    label_volume = rawfile.reshape(shape_raw)
    label_volume = label_volume.transpose([1, 2, 0])

    # # raw seg image is upside-down: do vertical flip
    # label_volume = np.flipud(label_volume)

    return label_volume


def load_datapair_lookup_table(datapath):
    pair_list = []
    with open(datapath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            image_name, mask_name, phase_name = line.split()
            pair_list.append([image_name, mask_name, phase_name])
    return pair_list


def load_year1_extended_dataset(pair_list, datapath, num_to_load=None):
    dicom_path = "dicom_image"
    mask_path = "segmentation_mask"

    if num_to_load is not None:
        load_counter = 0

    # master list that will have shape [subjects, 4, max(z), 512, 512]
    dicom_masterlist = []
    # and [subjects, (# of masks i.e. lesions), z, 512, 512]
    mask_masterlist = []
    # and ids
    ids_masterlist = []

    # traverse through subfolders and phases
    list_phase = ['A', 'D', 'P', 'Pre']
    for i in range(len(pair_list)):
        dicom_name, mask_name, phase_name = pair_list[i]

        # 4phase dicom loading loop
        dicom_img_4phase = []
        for phase in list_phase:
            img_path = os.path.join(datapath, dicom_path, dicom_name, phase)
            # load dicom image and append to 4phase list
            dicom_img = read_dicom_series(img_path, filepattern=phase + "_*")
            dicom_img_4phase.append(dicom_img)

        # mask loading step: the name is usually like A1_A.raw
        # but if multiple masks (lesions) exist, it's like A109_P1.raw, A109_P2.raw
        mask_list = []
        mask_base = str(mask_name + "_" + phase_name)
        # loop over all masks, and get the mask that matches the name
        mask_found = False
        mask_found_list = []
        try:
            for mask in os.listdir(os.path.join(datapath, mask_path)):
                if mask.startswith(mask_base):
                    mask_found = True
                    mask_found_list.append(mask)
                    mask_raw = read_liver_seg_masks_raw_year1_extended(os.path.join(datapath, mask_path, mask))
                    mask_list.append(mask_raw)
            if not mask_found:
                raise FileNotFoundError
            print("found dicom & masks pair: {} {}".format(dicom_name, mask_found_list))
        except FileNotFoundError:
            print("WARNING: dicom & masks pair: {} {} NOT FOUND. skipped loading. ".format(dicom_name, mask_found_list))
            continue

        # if both dicom image and masks are found, append them to masterlist
        dicom_masterlist.append(dicom_img_4phase)
        mask_masterlist.append(mask_list)
        ids_masterlist.append([dicom_name, mask_base])

        if num_to_load is not None:
            load_counter += 1
            if load_counter == num_to_load:
                print("num_to_load reached. early stopping loading")
                break

    return dicom_masterlist, mask_masterlist, ids_masterlist


def printout_to_jpg(images, masks, ids, printout_path):
    print("printout started!")
    assert len(images) == len(masks)
    for i in range(len(images)):
        image, mask, id = images[i], masks[i], ids[i]
        dicom_name, mask_phasename = id[0], id[1][-1:]
        print("printing {}...".format(dicom_name))
        if not os.path.exists(os.path.join(printout_path, str(dicom_name))):
            os.mkdir(os.path.join(printout_path, str(dicom_name)))

        # take z_max from 4-phase dicom images
        z_max = max(image[0].shape[2], image[1].shape[2],
                    image[2].shape[2], image[3].shape[2])
        z_list = [image[0].shape[2], image[1].shape[2], image[2].shape[2], image[3].shape[2]]
        # assert all z are identical, if not, raise warning
        try:
            assert z_list.count(z_list[0]) == len(z_list)
        except AssertionError:
            print("WARNING: {} 4-phase has non-matching number of slices".format(dicom_name))

        # mask may or may not align with the 4-phase dicom images
        # since we have mask_phasename, align the mask to that specific phase
        # always pasting the mask from the first slice is always "correct"
        # first, assert all z of the (possibly) multiple masks are the same
        z_mask = [mask_.shape[2] for mask_ in mask]
        assert z_mask.count(z_mask[0]) == len(z_mask)

        # then, add all the masks to have a single mask tensor
        if len(mask) == 1:
            mask = mask[0]
        else:
            mask = np.sum(mask, axis=0)
            # assert it is still binary
            try:
                assert np.array_equal(mask, mask.astype(bool))
            except AssertionError:
                print("WARNING: {} has overlapping lesions. merging them to binary...".format(dicom_name))
                mask = np.where(mask > 0, 1, 0)

        for i_slice in range(z_max):
            slice_path = os.path.join(printout_path, str(dicom_name), str(i_slice) + '.jpg')
            cat_slice = np.zeros([512, 512 * 5])
            for i_phase in range(4):
                try:
                    slice = np.copy(image[i_phase][:, :, i_slice])
                except IndexError:
                    slice = np.random.randint(0, 255, size=(512, 512))
                slice_norm = (slice - float(np.amin(slice))) / (
                        float(np.amax(slice)) - float(np.amin(slice)))
                slice_norm *= 255
                cat_slice[:, 512 * i_phase:512 * i_phase + 512] = slice_norm
            # finally attach mask at the end
            try:
                slice_mask = np.copy(mask[:, :, i_slice])
            except IndexError:
                slice_mask = np.random.randint(0, 1, size=(512, 512))
            slice_mask *= 255
            cat_slice[:, 512 * 4:] = slice_mask

            imsave(slice_path, cat_slice)

    print("test printout")
    return


if __name__ == "__main__":
    # load data pair lookup table that looks like
    # image_name    mask_name   phase_name (that the doctors annotated from)
    """
    HCC_1106	A1	A
    HCC_1107	A20	A
    HCC_1108	A4	A
    HCC_1114	A22	A
    """
    # lookup table & data locations
    lookup_table_location = "/home/tkdrlf9202/Datasets/liver_year1_dataset_extended_1809/image_mask_pair_list.txt"
    data_location = "/home/tkdrlf9202/Datasets/liver_year1_dataset_extended_1809"
    printout_path = "/home/tkdrlf9202/Datasets/liver_year1_dataset_extended_1809/sanity_check"

    pair_list = load_datapair_lookup_table(lookup_table_location)
    images, masks, ids = load_year1_extended_dataset(pair_list, data_location, num_to_load=None)
    printout_to_jpg(images, masks, ids, printout_path)
