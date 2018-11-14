# used for converting custom dataset to have consistent data sturcture

# CT: each subjects have 4 phases: A, D, P, Pre
# each phases have dicom slices with variable height: A_0000.DCM ~ A_0xxx.DCM

# roi_image: same subjects, same 4 phases
# each phases have dicom slices that contains only box-labeld image (others mot included)

# target: make roi_coordinate: same 4 phases per subject, create coordinates if exist
# otherwise make empty text

import numpy as np
import os
import glob

data_path = '/home/tkdrlf9202/Datasets/liver_year1_dataset_extended_1809/dicom_image'
#ct_path = os.path.join(data_path, 'ct')
# year1 extended data hack, 181031
ct_path = data_path
roi_image_path = os.path.join(data_path, 'roi_image')

# rename CT dataset phases to have consistent name (A, D, P, Pre)
# fix V -> P, LA -> A
# traverse over subjects
subject_name_list = []
for subject in glob.glob(os.path.join(ct_path, '*')):
    # append basename of subject to compare with roi_image
    subject_name_list.append(os.path.basename(os.path.normpath(subject)))
    for subfolder in glob.glob(os.path.join(subject, '*')):
        # retrieve basename of the path
        basename = os.path.basename(os.path.normpath(subfolder))
        # is the basename is V or LA, fix it
        if basename == 'V':
            for image in glob.glob(os.path.join(subfolder, '*')):
                # get suffix and fix basename (ex: 0012.DCM)
                suffix = image[-8:]
                basename_fixed = 'P_' + suffix
                # make path for renaming
                image_fixed = os.path.join(subfolder, basename_fixed)
                # rename image
                os.rename(image, image_fixed)
            # rename folder
            os.rename(subfolder, os.path.join(subject, 'P'))

        elif basename == 'LA':
            for image in glob.glob(os.path.join(subfolder, '*')):
                # get suffix and fix basename (ex: 0012.DCM)
                suffix = image[-8:]
                basename_fixed = 'A_' + suffix
                # make path for renaming
                image_fixed = os.path.join(subfolder, basename_fixed)
                # rename image
                os.rename(image, image_fixed)
            # rename folder
            os.rename(subfolder, os.path.join(subject, 'A'))
# sort the name list to compare with roi_image folders
subject_name_list.sort()

###################### end the script for just fixing folder names, 1801031
print("folder name fixing ended")

subject_name_list_roi = []
# check roi_image data integrity
for subject in glob.glob(os.path.join(roi_image_path, '*')):
    subject_name_list_roi.append(os.path.basename(os.path.normpath(subject)))
    for subfolder in glob.glob(os.path.join(subject, '*')):
        basename = os.path.basename(os.path.normpath(subfolder))
        if basename not in ['A', 'D', 'P', 'Pre']:
            print('error: phase name ' + str(basename) + ' not in list [A, D, P, Pre], fixing...')
            if basename == 'V':
                for image in glob.glob(os.path.join(subfolder, '*')):
                    # get suffix and fix basename (ex: 0012.DCM)
                    suffix = image[-8:]
                    basename_fixed = 'P_' + suffix
                    # make path for renaming
                    image_fixed = os.path.join(subfolder, basename_fixed)
                    # rename image
                    os.rename(image, image_fixed)
                # rename folder
                os.rename(subfolder, os.path.join(subject, 'P'))

            elif basename == 'LA':
                for image in glob.glob(os.path.join(subfolder, '*')):
                    # get suffix and fix basename (ex: 0012.DCM)
                    suffix = image[-8:]
                    basename_fixed = 'A_' + suffix
                    # make path for renaming
                    image_fixed = os.path.join(subfolder, basename_fixed)
                    # rename image
                    os.rename(image, image_fixed)
                # rename folder
                os.rename(subfolder, os.path.join(subject, 'A'))

subject_name_list_roi.sort()
if subject_name_list != subject_name_list_roi:
    print('error: subject name between ct and roi_image does not match')
    difference = set(subject_name_list).symmetric_difference(set(subject_name_list_roi))
    print('mismatching subjects: ' + str(difference))
