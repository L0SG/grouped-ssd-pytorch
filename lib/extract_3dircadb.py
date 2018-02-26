import numpy as np
IMG_DTYPE = np.float
SEG_DTYPE = np.uint8

import dicom
import natsort
import glob, os
import re
import h5py


def read_dicom_series(directory, filepattern="image_*"):
    """ Reads a DICOM Series files in the given directory.
    Only filesnames matching filepattern will be considered"""

    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise ValueError("Given directory does not exist or is a file : " + str(directory))
    print
    '\tRead Dicom', directory
    lstFilesDCM = natsort.natsorted(glob.glob(os.path.join(directory, filepattern)))
    print
    '\tLength dicom series', len(lstFilesDCM)
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


def read_liver_lesion_masks(masks_dirname):
    """Since 3DIRCAD provides an individual mask for each tissue type (in DICOM series format),
    we merge multiple tissue types into one Tumor mask, and merge this mask with the liver mask

    Args:
        masks_dirname : MASKS_DICOM directory containing multiple DICOM series directories,
                        one for each labelled mask
    Returns:
        numpy array with 0's for background pixels, 1's for liver pixels and 2's for tumor pixels
    """
    tumor_volume = None
    liver_volume = None

    # For each relevant organ in the current volume
    for organ in os.listdir(masks_dirname):
        organ_path = os.path.join(masks_dirname ,organ)
        if not os.path.isdir(organ_path):
            continue

        organ = organ.lower()

        if organ.startswith("livertumor") or re.match("liver.yst.*", organ) or organ.startswith("stone") or organ.startswith("metastasecto"):
            print('Organ' ,masks_dirname ,organ)
            current_tumor = read_dicom_series(organ_path)
            current_tumor = np.clip(current_tumor ,0 ,1)
            # Merge different tumor masks into a single mask volume
            tumor_volume = current_tumor if tumor_volume is None else np.logical_or(tumor_volume ,current_tumor)
        elif organ == 'liver':
            print('Organ' ,masks_dirname ,organ)
            liver_volume = read_dicom_series(organ_path)
            liver_volume = np.clip(liver_volume, 0, 1)

    label_volume = np.zeros(liver_volume.shape)
    label_volume[tumor_volume == 1 ] = 1
    return label_volume


def step1_preprocess_img_slice(img_slc):
    """
    Preprocesses the image 3d volumes by performing the following :
    1- Rotate the input volume so the the liver is on the left, spine is at the bottom of the image
    2- Set pixels with hounsfield value great than 1200, to zero.
    3- Clip all hounsfield values to the range [-100, 400]
    4- Normalize values to [0, 1]

    Args:
        img_slc: raw image slice
    Return:
        Preprocessed image slice
    """
    img_slc = img_slc.astype(IMG_DTYPE)
    img_slc[img_slc > 1200] = 0
    img_slc = np.clip(img_slc, -100, 400)
    img_slc = normalize_image(img_slc)

    return img_slc


def normalize_image(img):
    """ Normalize image values to [0,1] """
    min_, max_ = float(np.min(img)), float(np.max(img))
    return (img - min_) / (max_ - min_)


img = read_dicom_series("test_image/3Dircadb1.17/PATIENT_DICOM/")
lbl = read_liver_lesion_masks("test_image/3Dircadb1.17/MASKS_DICOM/")
for idx in range(img.shape[2]):
    img[:, :, idx] = step1_preprocess_img_slice(img[:, :, idx])

img = np.transpose(img, (2, 0, 1))
lbl = np.transpose(lbl, (2, 0, 1))

# stack 3 images
img_3slices = []
lbl_3slices = []
for idx in range(1, img.shape[0]-1):
    img_3slices.append(img[idx-1:idx+2, :, :])
    lbl_3slices.append(lbl[idx-1:idx+2, :, :])

img_3slices_4phase = [img_3slices] * 4
lbl_3slices_4phase = [lbl_3slices] * 4

img_3slices_4phase = np.array(img_3slices_4phase)
lbl_3slices_4phase = np.array(lbl_3slices_4phase)


dataset_save_location = '/home/tkdrlf9202/Datasets/liver_lesion_aligned/3dircadb_dataset_ponly_aligned.h5'

print('dumping dataset...')
with h5py.File(dataset_save_location, 'w') as dump_h5:
    group_ct = dump_h5.create_group('ct')
    group_coordinate = dump_h5.create_group('label')
    for idx in range(1):
        group_ct.create_dataset('ct_' + str(idx), data=img_3slices_4phase)
        group_coordinate.create_dataset('coordinate_' + str(idx), data=lbl_3slices_4phase)
    dump_h5.close()