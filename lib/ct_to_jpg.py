import numpy as np
import glob
import os
import natsort
import pydicom as dicom
import scipy


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


def ct_to_jpg(data_path, output_jpg_path, num_data_to_load):
    """
    print-out the multi-phase dicom CT file to concatenated jpg
    with input dicom of 4 phase of 512 x 512, each output jpg size is 512 x (512 x 4) wide
    if not z-aligned, the missing phase image is replaced with static noise indicating that it's missing
    """

    # get list of subfolder names
    dir_list = os.listdir(data_path)
    dir_list = sorted(dir_list)

    count_for_early_stop = 0

    # traverse through subfolders
    list_phase = ['A', 'D', 'P', 'Pre']
    for dir_name in dir_list:
        # obtain absolute path of the subject
        path_subject = os.path.join(data_path, dir_name)
        print("processing " + str(path_subject))
        # load dicom image of the subject
        dicom_image = []
        for phase in list_phase:
            dicom_image.append(read_dicom_series(os.path.join(path_subject, phase), phase + "_*"))
        dicom_image = dicom_image[0], dicom_image[1], dicom_image[2], dicom_image[3]

        # save dicom data as image
        mask_image_save_path = output_jpg_path
        if not os.path.exists(os.path.join(mask_image_save_path, str(dir_name))):
            os.makedirs(os.path.join(mask_image_save_path, str(dir_name)))

        # take the maximum one: for outputting concatenated image
        z_max = max(dicom_image[0].shape[2], dicom_image[1].shape[2],
                    dicom_image[2].shape[2], dicom_image[3].shape[2])

        for idx in range(z_max):
            cat_slice_save_path = os.path.join(mask_image_save_path, str(dir_name), str(idx) + '.jpg')
            cat_slice = np.zeros([512, 512 * 4])

            for idx_phase in range(4):
                try:
                    dicom_slice = np.copy(dicom_image[idx_phase][:, :, idx])
                except IndexError:
                    dicom_slice = np.random.randint(0, 255, size=(512, 512))
                dicom_slice_normalized = (dicom_slice - float(np.amin(dicom_slice))) / (
                        float(np.amax(dicom_slice)) - float(np.amin(dicom_slice)))
                dicom_slice_normalized *= 255

                cat_slice[:, 512 * idx_phase:512 * idx_phase + 512] = dicom_slice_normalized

            scipy.misc.imsave(cat_slice_save_path, cat_slice)

        # partial load defined in num_data_to_load
        count_for_early_stop += 1
        if num_data_to_load is None:
            continue
        if count_for_early_stop == num_data_to_load:
            break
            
    return


if __name__ == '__main__':
    data_path = "/media/hdd/tkdrlf9202/Datasets/liver_lesion_structured_noalign/ct"
    output_jpg_path = "/media/hdd/tkdrlf9202/Datasets/liver_lesion_structured_noalign_jpg"

    ct_to_jpg(data_path, output_jpg_path, None)
    print("done")


