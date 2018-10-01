import os
import numpy as np
import nibabel as nib
from nibabel.orientations import flip_axis
import scipy.misc

img = nib.load('/home/tkdrlf9202/Datasets/snuh_HCC_sample_1807/nii_DEEPHI/HCC_1106/HCC_1106.nii')

#%% get tensor data
data = img.get_fdata()

# it's rotated 90 to the right
# get it back to original dicom orientation: rotate 90 to the left
data = np.rot90(data)

# and it's upside-down, get it back
data = np.flipud(data)

#%% try to save all slices of the data: is it valid?
sanitycheck_datapath = 'sanitycheck_nii_label'

if not os.path.exists(sanitycheck_datapath):
    os.makedirs(sanitycheck_datapath)
for idx_slice in range(data.shape[2]):
    data_slice = data[:, :, idx_slice]
    scipy.misc.imsave(os.path.join(sanitycheck_datapath, 'label_' + str(idx_slice) + '.jpg'), data_slice)

