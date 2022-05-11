# sanity checker script for year1 extended dataset received in 1904
# modified based on 1809 script

import numpy as np
import glob
import os
import natsort
import pydicom as dicom
import scipy
from imageio import imwrite
import lib.mask2bbox as mask2bbox
import cv2
import multiprocessing


def extract_metadata_from_excel():
    import pandas as pd
    import math

    excel_path = "/media/hdd/tkdrlf9202/Datasets/liver_year1_dataset_extended_1904_preprocessed/metadata_preprocessed_190531.xlsx"
    output_path = "/media/hdd/tkdrlf9202/Datasets/liver_year1_dataset_extended_1904_preprocessed/metadata_preprocessed_190531.txt"

    df = pd.read_excel(excel_path)

    # get names deltas for each phase
    ids = df['Index_New']
    # deltas for A, D, P, Pre
    d_a = df['A']
    d_d = df['D']
    d_p = df['P']
    d_pre = df['Pre']
    # MEDIP pivot phase
    pivot = df['MEDIP']

    ids = ids.tolist()[1:]
    d_a = d_a.tolist()[1:]
    d_d = d_d.tolist()[1:]
    d_p = d_p.tolist()[1:]
    d_pre = d_pre.tolist()[1:]
    pivot = pivot.tolist()[1:]
    lengths = [len(ids), len(d_a), len(d_d), len(d_p), len(d_pre), len(pivot)]
    assert len(set(lengths)) == 1

    d_list = [d_a, d_d, d_p, d_pre]
    # fix broken values
    for d_sublist in d_list:
        for i in range(len(d_sublist)):
            if type(d_sublist[i]) == str:
                d_sublist[i] = int(d_sublist[i])
            if math.isnan(d_sublist[i]):
                d_sublist[i] = 0

    # #%% doctors used only one of the phase as the "pivot". assert that it holds
    # import numpy as np
    # deltas = np.array(d_list).transpose()
    # for i_row in range(deltas.shape[0]):
    #     delta = deltas[i_row]
    #     assert np.count_nonzero(delta) == 1

    with open(output_path, 'w') as f:
        header = 'ID\tA\tD\tP\tPre\tMEDIP\n'
        f.write(header)
        for i in range(len(ids)):
            line = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(ids[i], d_a[i], d_d[i], d_p[i], d_pre[i], pivot[i])
            f.write(line)

    print("done")


def merge_mask_metadata_from_1809():
    lookup_table_location = "/media/hdd/tkdrlf9202/Datasets/liver_year1_dataset_extended_1904_preprocessed/metadata_before_final/image_mask_pair_list_1809.txt"
    metadata_location = "/media/hdd/tkdrlf9202/Datasets/liver_year1_dataset_extended_1904_preprocessed/metadata_before_final/metadata_preprocessed_190531.txt"
    metadata_final_location = "/media/hdd/tkdrlf9202/Datasets/liver_year1_dataset_extended_1904_preprocessed/metadata.txt"

    with open(lookup_table_location, 'r') as f:
        lookup = f.readlines()
    lookup = [line.strip().split() for line in lookup]

    with open(metadata_location, 'r') as f:
        metadata = f.readlines()[1:]
    metadata = [line.strip().split() for line in metadata]

    # make dict for lookup data like key: HCC_1106 & value: (A1, A)
    lookup_dict = dict()
    for i in range(len(lookup)):
        lookup_dict[lookup[i][0]] = (lookup[i][1], lookup[i][2])

    # query the subject id from metadata to the lookup dict
    for i in range(len(metadata)):
        mask_id, mask_phase = lookup_dict[metadata[i][0]]
        mask_phase_from_metadata = metadata[i][5]
        assert mask_phase == mask_phase_from_metadata,\
            "mask phase from 1809 dataset does not match the 1904 annotation. something is wrong..."
        metadata[i].append(mask_id)

    # save the final metadata
    with open(metadata_final_location, 'w') as f:
        header = 'ID\tDelta_A\tDelta_D\tDelta_P\tDelta_Pre\tMask_Phase\tMask_ID\n'
        f.write(header)
        for i in range(len(metadata)):
            f.write('\t'.join(metadata[i]) + '\n')
    print("all done")


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


def load_metadata(datapath):
    pair_list = []
    with open(datapath, 'r') as f:
        # ditch header
        lines = f.readlines()[1:]
        for line in lines:
            elements = line.split()
            pair_list.append(elements)
    return pair_list


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
    # edge edge case: HCC_1237 Pre phase (-3024~1166)
    if np.amin(img) == -3024 and np.amax(img) == 1166:
        img = np.clip(img, -1024, 1166)

    # for handling unusual cases where min val is -2000 (outside the organ) and 0~2400 range inside the organ
    # ex: HCC_1446 from 1904 set
    elif np.amin(img) == -2000 or np.amin(img) == -2048:
        img[img == -2000] = 0
        img[img == -2048] = 0
    # ex2: HCC_1224 has reallly unusual -2019~4100 range... wat?
    elif np.amin(img) < -2000:
        print("WARNING: HU value range of this subject is extremely unusual. double-check the correctness of windowing.")
        img[img < 0] = 0

    #if np.amin(img) == 0:
    if np.mean(img) > 0:
        # print("WARNING: This CT image has minimum CT HU of 0. Double check if the windowing function is right")
        # print("WARNING: manually adjusting HU to have minimum value of -1024...")
        img = np.subtract(img, 1024)

    # # if "properly" preprocessed, the mean pixel value should be around -500
    # assert np.mean(img) < -200,\
    #     "mean pixel value {} is not around the assumed ~-500 range. double-check the correctness of apply_window.".format(np.mean(img))

    img[img > 1200] = 0

    upper_grey = window_level + (window_width / 2.)
    lower_grey = window_level - (window_width / 2.)

    img_windowed = np.clip(img, lower_grey, upper_grey)

    return img_windowed


def load_year1_extended_dataset(pair_list, datapath, i, img_size):
    dicom_path = "dicom_image"
    mask_path = "segmentation_mask"

    # master list that will have shape [subjects, 4, max(z), 512, 512]
    dicom_masterlist = []
    # and [subjects, (# of masks i.e. lesions), z, 512, 512]
    mask_masterlist = []
    # additionally merged masks (if there's multiples lesions), so [subjects, z, 512, 512]
    mask_merged_masterlist = []
    # and ids
    ids_masterlist = []

    # traverse through subfolders and phases
    list_phase = ['A', 'D', 'P', 'Pre']

    dicom_name, delta_a, delta_d, delta_p, delta_pre, phase_name, mask_name = pair_list[i]
    # make dict for deltas for mask post-processing
    deltas = dict()
    deltas['A'], deltas['D'], deltas['P'], deltas['Pre'] = int(delta_a), int(delta_d), int(delta_p), int(delta_pre)

    # check if the dicom data exists
    # if not, it means that data is excluded from previous preprocessing
    # ex: HCC_1200 and HCC_1266 from 1904 data had wrong phases
    if not os.path.exists(os.path.join(datapath, dicom_path, dicom_name)):
        print("WARNING: {} not found in {}. skipping...".format(dicom_name, os.path.join(datapath, dicom_path)))
        return None, None, None, None

    # 4phase dicom loading loop
    dicom_img_4phase = []
    for phase in list_phase:
        img_path = os.path.join(datapath, dicom_path, dicom_name, phase)
        # load dicom image and append to 4phase list
        dicom_img = read_dicom_series(img_path, filepattern=phase + "_*").astype(np.int)
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
                # NEW in 1904 script: post-process mask_raw given delta metadata
                delta_mask = deltas[phase_name]
                # cut out mask file following delta of the pivot phase
                size_pivot_phase = dicom_img_4phase[list_phase.index(phase_name)].shape[2]
                mask_cut = mask_raw[:, :, delta_mask:delta_mask+size_pivot_phase]
                assert mask_cut.shape[2] == size_pivot_phase
                mask_list.append(mask_cut)
        if not mask_found:
            raise FileNotFoundError

        # preprocess dicom images
        max_pre, min_pre, mean_pre = np.amax(dicom_img_4phase), np.amin(dicom_img_4phase), np.mean(dicom_img_4phase)
        dicom_img_4phase = window_and_normalize(dicom_img_4phase)
        max_pro, min_pro, mean_pro = np.amax(dicom_img_4phase), np.amin(dicom_img_4phase), np.mean(dicom_img_4phase)

        print("found dicom & masks pair: {} {}, max_pre {} min_pre {} mean_pre {} max_pro {} min_pro {} mean_pro {}".
              format(dicom_name, mask_found_list,
                     max_pre, min_pre, mean_pre, max_pro, min_pro, mean_pro))

    except FileNotFoundError:
        print("WARNING: dicom & masks pair: {} {} NOT FOUND. skipped loading. ".format(dicom_name, mask_found_list))
        return None, None, None

    # if both dicom image and masks are found, append them to masterlist
    # BUT, resize if specified
    if img_size != 512:
        dicom_img_4phase = [cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_BICUBIC) for img in dicom_img_4phase]
        mask_list = [cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST) for img in mask_list]

    dicom_masterlist.append(dicom_img_4phase)
    mask_masterlist.append(mask_list)
    ids_masterlist.append([dicom_name, mask_base])

    # merge mask_list by summing them all (then cast all values >1 back to 1)
    mask_stacked = np.stack(mask_list)
    mask_merged = np.sum(mask_stacked, axis=0)
    mask_merged[mask_merged > 1] = 1
    # assert binary
    assert np.array_equal(mask_merged, mask_merged.astype(bool))
    mask_merged = mask_merged.astype(np.uint8)
    mask_merged_masterlist.append(mask_merged)

    return dicom_masterlist, mask_masterlist, mask_merged_masterlist, ids_masterlist

def window_and_normalize(image):
    # apply HU window and normalize to [0, 1]
    for j in range(len(image)):
        # apply HU windowing to dicom image
        image[j] = apply_window(image[j], window_width=400, window_level=50).astype(np.float32)
        # normalize
        image[j] = (image[j] - float(np.amin(image[j]))) / (
                float(np.amax(image[j])) - float(np.amin(image[j])))

    return image


def printout_to_jpg(images, masks, ids, printout_path, img_size):
    assert len(images) == len(masks)
    for i in range(len(images)):
        image, mask, id = images[i], masks[i], ids[i]
        dicom_name, mask_phasename = id[0], id[1][-1:]
        print("printing {}...".format(dicom_name))
        if not os.path.exists(os.path.join(printout_path, str(dicom_name))):
            os.mkdir(os.path.join(printout_path, str(dicom_name)))

        for j in range(len(image)):
            # apply HU windowing to dicom image for better visuals
            image[j] = apply_window(image[j], window_width=400, window_level=50)

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
            cat_slice = np.zeros([img_size, img_size * 5])
            for i_phase in range(4):
                try:
                    slice = np.copy(image[i_phase][:, :, i_slice])
                except IndexError:
                    slice = np.random.randint(0, 255, size=(img_size, img_size))
                slice_norm = (slice - float(np.amin(slice))) / (
                        float(np.amax(slice)) - float(np.amin(slice)))
                slice_norm *= 255
                cat_slice[:, img_size * i_phase:img_size * i_phase + img_size] = slice_norm
            # finally attach mask at the end
            try:
                slice_mask = np.copy(mask[:, :, i_slice])
            except IndexError:
                slice_mask = np.random.randint(0, 1, size=(img_size, img_size))
            slice_mask *= 255
            cat_slice[:, img_size * 4:] = slice_mask

            imwrite(slice_path, cat_slice.astype(np.uint8))

    return



def dataset_creation_loop(metadata):
    metadata_ml_ready_result = []
    for i in range(len(metadata)):
        # if metadata[i][0] != "HCC_1237":
        #     continue
        images, masks, masks_merged, ids = load_year1_extended_dataset(metadata, data_path, i, img_size)
        if images is None:
            continue
        # the above method assumed list (subject-level) of data but now we call per subject
        # remove the outermost list which has len 1
        images, masks, masks_merged, ids = images[0], masks[0], masks_merged[0], ids[0]

        # additionally get bboxes from masks for GSSD
        # masks also gets smoothed inside this method, retrieve them too
        if masks_merged is not None:
            image_final, mask_final, bbox_final = mask2bbox.convert(images, masks_merged, ids, img_size,
                                                    debug_print_path=debug_print_path)

        # ad-hoc creation of phase label (that doctors annotated the images from)
        phase_to_token = {'A': 0, 'D': 1, 'P': 2, 'Pre': 3}
        token = phase_to_token[metadata[i][5]]

        # finally save the final datapoint to disk
        subject_name = ids[0]
        output_path_subject = os.path.join(output_path, subject_name)
        if not os.path.isdir(output_path_subject):
            os.makedirs(output_path_subject, exist_ok=True)

        for i in range(len(image_final)):
            datapoint_name_relative = subject_name + '_' + str(i)
            datapoint_name_ct = datapoint_name_relative + '_ct.npy'
            datapoint_name_mask = datapoint_name_relative + '_mask.npy'
            datapoint_name_bbox = datapoint_name_relative + '_bbox.npy'
            datapoint_name_phase = datapoint_name_relative + '_phase.npy'
            np.save(os.path.join(output_path_subject, datapoint_name_ct), image_final[i])
            np.save(os.path.join(output_path_subject, datapoint_name_mask), mask_final[i])
            np.save(os.path.join(output_path_subject, datapoint_name_bbox), bbox_final[i])
            # phase label is same for all images, just repeat save for ease of use
            np.save(os.path.join(output_path_subject, datapoint_name_phase), token)
            metadata_line = os.path.join(subject_name, datapoint_name_relative) + '|' + subject_name
            metadata_ml_ready_result.append(metadata_line)
    return metadata_ml_ready_result



if __name__ == "__main__":
    # load metadata that looks like
    # image_name    deltas   mask_phase (that the doctors annotated from), mask_id
    """
    ID	Delta_A	Delta_D	Delta_P	Delta_Pre	Mask_Phase	Mask_ID
    HCC_1104	0	0	10	0	P	A5
    HCC_1105	0	14	0	0	D	A9
    HCC_1106	6	0	0	0	A	A1
    HCC_1107	19	0	0	0	A	A20
    HCC_1111	0	6	0	0	D	A8
    HCC_1112	0	10	0	0	D	A25
    HCC_1114	19	0	0	0	A	A22
    HCC_1117	3	0	0	0	A	A19
    """

    metadata_path = "/media/hdd/tkdrlf9202/Datasets/liver_year1_dataset_extended_1904_preprocessed/metadata.txt"
    data_path = "/media/hdd/tkdrlf9202/Datasets/liver_year1_dataset_extended_1904_preprocessed"
    printout_path = "/media/hdd/tkdrlf9202/Datasets/liver_year1_dataset_extended_1904_preprocessed/sanity_check"
    output_path = "/media/hdd/tkdrlf9202/Datasets/liver_year1_dataset_extended_1904_preprocessed/output"
    debug_print_path = "/media/hdd/tkdrlf9202/Datasets/liver_year1_dataset_extended_1904_preprocessed/debug_print"

    # path for saving ML-ready cooked data
    img_size = 512
    output_path = "/home/tkdrlf9202/Datasets/liver_year1_dataset_extended_1904_preprocessed/ml_ready_phaselabel"
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    metadata = load_metadata(metadata_path)

    metadata_ml_ready = open(os.path.join(output_path, 'metadata.txt'), 'w+')

    # # debug, non-parallel version
    # results = dataset_creation_loop(metadata)

    # parallel
    N = 16
    with multiprocessing.Pool(processes=N) as p:
        results = p.map(dataset_creation_loop, [[meta] for meta in metadata])
        p.close()
        p.join()
    results = [item for sublist in results for item in sublist]

    for i in range(len(results)):
        metadata_ml_ready.write(results[i] + '\n')

    # load and printout one subject at a time (suppress RAM usage)
    # for i in range(len(metadata)):
    #     # if metadata[i][0] != "HCC_1237":
    #     #     continue
    #     images, masks, masks_merged, ids = load_year1_extended_dataset(metadata, data_path, i, img_size)
    #     if images is None:
    #         continue
    #     # the above method assumed list (subject-level) of data but now we call per subject
    #     # remove the outermost list which has len 1
    #     images, masks, masks_merged, ids = images[0], masks[0], masks_merged[0], ids[0]
    #
    #     # additionally get bboxes from masks for GSSD
    #     # masks also gets smoothed inside this method, retrieve them too
    #     if masks_merged is not None:
    #         image_final, mask_final, bbox_final = mask2bbox.convert(images, masks_merged, ids, img_size,
    #                                                 debug_print_path=debug_print_path)
    #
    #     # finally save the final datapoint to disk
    #     subject_name = ids[0]
    #     output_path_subject = os.path.join(output_path, subject_name)
    #     if not os.path.isdir(output_path_subject):
    #         os.makedirs(output_path_subject, exist_ok=True)
    #
    #     for i in range(len(image_final)):
    #         datapoint_name_relative = subject_name + '_' + str(i)
    #         datapoint_name_ct = datapoint_name_relative + '_ct.npy'
    #         datapoint_name_mask = datapoint_name_relative + '_mask.npy'
    #         datapoint_name_bbox = datapoint_name_relative + '_bbox.npy'
    #         np.save(os.path.join(output_path_subject, datapoint_name_ct), image_final[i])
    #         np.save(os.path.join(output_path_subject, datapoint_name_mask), mask_final[i])
    #         np.save(os.path.join(output_path_subject, datapoint_name_bbox), bbox_final[i])
    #         metadata_line = os.path.join(subject_name, datapoint_name_relative) + '|' + subject_name
    #         metadata_ml_ready.write(metadata_line + '\n')
        # if images is not None:
        #     printout_to_jpg(images, masks, ids, printout_path)