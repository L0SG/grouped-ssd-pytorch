# https://www.kaggle.com/voglinio/from-masks-to-bounding-boxes
import skimage
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from imageio import imwrite

# following previous labeling method
lesion_class_label = 0

def convert(images, masks, ids, img_size, debug_print_path=None):
    # convert mask to bboxes
    bboxes = []
    # we will also gather gaussian-smoothed masks (since these are actually ml-usable)
    masks_smoothed = []

    mask = masks
    subject_name = ids[0]
    phase_name = ids[1][-1]
    phase_dict = {'A':0, 'D':1, 'P':2, 'Pre':3}
    # get image that corresponds to annotation
    image = images[phase_dict[phase_name]]
    # stacked 4-phase image for the reference in the future
    image_stacked = np.stack(images)

    for idx_slice in range(mask.shape[2]):
        mask_slice = mask[:, :, idx_slice]
        image_slice = image[:, :, idx_slice]
        image_slice_4phase = np.stack(images, axis=0)[:, :, :, idx_slice]

        # if the slice has the mask label, there will be values of 1
        if mask_slice.max() == 1:
            # but for the very edge of the annotations, there will be noise-like few pixel masks (not actually useful..)
            # so let's just filter mask that has less than some pixels...
            if np.count_nonzero(mask_slice == 1) < 1:
                print("INFO: too few number of pixel masks for {}th slice. skipped".format(idx_slice))
                bboxes.append(None)
                masks_smoothed.append(None)
                continue
            """
            # TODO: remove after comparsion 190709
            #  compare blurred version vs non-blurred version
            mask_nonblurred = mask_slice.copy()
            # # filter out super-tiny noise
            # kernel_open = np.ones((5, 5), np.uint8)
            # mask_filtered = cv2.morphologyEx(mask_nonblurred, cv2.MORPH_OPEN, kernel_open)
            # # fill small holes
            # kernel_close = np.ones((5, 5), np.uint8)
            # mask_filtered = cv2.morphologyEx(mask_filtered, cv2.MORPH_CLOSE, kernel_close)
            mask_final = mask_nonblurred
            mask_contour, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # try to get bboxes
            label_slice = label(mask_final)
            props = regionprops(label_slice)
            # draw
            image_print_nonblur = image_slice.copy()
            image_print_nonblur = np.stack((image_print_nonblur,) * 3, -1)
            cv2.drawContours(image_print_nonblur, mask_contour, -1, (0, 0, 1))
            for prop in props:
                # print("bbox", prop.bbox)
                cv2.rectangle(image_print_nonblur, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (1, 0, 0), 1)
            """

            # our seg mask is fuzzy: we need to get clean & continuous contour to gea a new clean mask
            # first, apply gaussian blur to try to smear the fuzzy mask
            mask_blurred = cv2.GaussianBlur(mask_slice, (11, 11), 0)
            """
            # if there is too small mask in the blurred image, it will not be a useful signal
            if np.count_nonzero(mask_blurred == 1) < 1:
                print("INFO: no mask found after blurring for {}th slice. skipped".format(idx_slice))
                continue
            """
            # filter out super-tiny noise
            kernel_open = np.ones((5, 5), np.uint8)
            mask_filtered = cv2.morphologyEx(mask_blurred, cv2.MORPH_OPEN, kernel_open)

            # fill small holes
            kernel_close = np.ones((5, 5), np.uint8)
            mask_filtered = cv2.morphologyEx(mask_filtered, cv2.MORPH_CLOSE, kernel_close)

            mask_final = mask_filtered

            mask_contour, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # plt.imshow(mask_slice)
            # plt.show()
            # plt.imshow(mask_blurred)
            # plt.show()

            # mask_print = mask_filtered.copy() * 255
            # mask_print = np.stack((mask_print,) * 3, -1)
            image_print = image_slice.copy()
            image_print = np.stack((image_print,) * 3, -1)
            image_print_4phase = image_slice_4phase.copy()
            image_print_4phase = np.stack((image_print_4phase,) * 3, -1)


            # cv2.drawContours(mask_print, mask_contour, -1, (0, 255, 0))
            cv2.drawContours(image_print, mask_contour, -1, (0, 1, 0), 2)
            for i_p in range(image_print_4phase.shape[0]):
                cv2.drawContours(image_print_4phase[i_p], mask_contour, -1, (0, 1, 0), 2)

            # try to get bboxes
            label_slice = label(mask_final)
            props = regionprops(label_slice)
            coordinate_list = []
            for prop in props:
                #print("bbox", prop.bbox)
                cv2.rectangle(image_print, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (1, 0, 0), 2)
                for i_p in range(image_print_4phase.shape[0]):
                    cv2.rectangle(image_print_4phase[i_p], (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (1, 0, 0), 2)
                # additionally construct bbox coordinates format matching miccai2018 notations
                x_start, y_start, x_end, y_end = prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]
                # use [x_min, y_min, x_max, y_max] type
                coordinate = [x_start, y_start, x_end, y_end]
                # append zero class label for lesion
                coordinate.append(lesion_class_label)
                coordinate_list.append(coordinate)

            # plt.imshow(image_print)
            # plt.show()
            # print("test")
            if coordinate_list != []:
                bboxes.append(coordinate_list)
                masks_smoothed.append(mask_final)
            else:
                bboxes.append(None)
                masks_smoothed.append(None)



            """
            # TODO: remove after comparsion 190709
            # concat two images for comparison
            image_print = np.concatenate([image_print, image_print_nonblur], axis=1)

            # plt.imshow(image_print)
            # plt.show()
            """

            if debug_print_path is not None:
                os.makedirs(os.path.join(debug_print_path,
                                         str(subject_name)), exist_ok=True)
                img_path = os.path.join(debug_print_path,
                                        str(subject_name), str(subject_name) +'_' + str(idx_slice) + '.jpg')

                imwrite(img_path, (image_print * 255).astype(np.uint8))
                os.makedirs(os.path.join(debug_print_path + '_4phase',
                                         str(subject_name)), exist_ok=True)
                img_4phase_path = os.path.join(debug_print_path + '_4phase',
                                               str(subject_name), str(subject_name) +'_' + str(idx_slice) + '.jpg')
                cat_slice = np.zeros([img_size, img_size * 4, 3])

                # NOTE: for printing, we swap the order of phases corresponding to "medical convention"
                # since we just sorted the image phase in alphabetical order (A, D, P, Pre)
                # Precontrast[3] -> Arterial[0] -> Portal[2] -> Delayed[1]
                medical_order = [3, 0, 2, 1]
                for idx_phase in range(image_print_4phase.shape[0]):
                    cat_slice[:, img_size * idx_phase:img_size * idx_phase + img_size, :] = image_print_4phase[medical_order[idx_phase]]

                imwrite(img_4phase_path, (cat_slice*255).astype(np.uint8))

        else:
            bboxes.append(None)
            masks_smoothed.append(None)

    # construct datapoints from the processed data
    assert len(masks_smoothed) == len(bboxes) == image_stacked.shape[3]
    image_final = []
    mask_final = []
    bbox_final = []

    # failsafe starting idx of 1 (since we will grab 3-slices image)
    for i in range(1, len(masks_smoothed)):
        if masks_smoothed[i] is not None:
            # i assumed [phase(4), channel(z-1:z+1), 512, 512] before
            image_cutout = np.transpose(image_stacked[:, :, :, (i-1):(i+2)], [0, 3, 1, 2])
            if image_cutout.shape[1] != 3:
                continue
            image_final.append(image_cutout)
            mask_final.append(masks_smoothed[i])
            bbox_final.append(bboxes[i])

    assert all([x is not None for x in image_final])
    assert all([x is not None for x in mask_final])
    assert all([x is not None for x in bbox_final])

    return image_final, mask_final, bbox_final