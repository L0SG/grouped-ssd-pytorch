from lib.utils import preprocess_liver_dataset
from lib.utils import load_liver_seg_dataset
import os
import h5py
import numpy as np

def load_liver_dataset(preprocessed_data_path, data_path):
    """
    loads custom liver dataset
    if preprocessed h5 data exists, load it
    if not, load and preprocess raw liver dataset
    :param preprocessed_data_path:
    :param data_path:
    :return: flattened CT and mask data
    """
    # check if the preprocessed dataset exists
    if os.path.isfile(preprocessed_data_path):
        # load the preprocessed dataset dump
        print('loading preprocessed dataset...')
        with h5py.File(preprocessed_data_path, 'r') as dataset_h5:
            ct_flattened = dataset_h5['ct'][:]
            mask_flattened = dataset_h5['mask'][:]
            dataset_h5.close()
    else:
        # load the liver dataset
        print('reading dicom and mask data...')
        list_ct, list_mask = load_liver_seg_dataset(data_path=data_path,
                                                    num_data_to_load=None)
        # preprocess the dataset
        print('preprocessing dataset...')
        list_ct_preprocessed, list_mask_preprocessed = preprocess_liver_dataset(list_ct, list_mask)

        # flatten the data from [subjects, width, height, depth] to [subject*depth(=datasize), 1(channel), width, height]
        ct_flattened = np.concatenate(list_ct_preprocessed, axis=2).transpose([2, 0, 1])
        ct_flattened = np.expand_dims(ct_flattened, axis=1)
        mask_flattened = np.concatenate(list_mask_preprocessed, axis=2).transpose([2, 0, 1])
        mask_flattened = np.expand_dims(mask_flattened, axis=1)

        # dump the preprocessed dataset for later use
        print('dumping dataset...')
        with h5py.File(preprocessed_data_path, 'w') as dump_h5:
            dump_h5.create_dataset('ct', data=ct_flattened)
            dump_h5.create_dataset('mask', data=mask_flattened)
            dump_h5.close()

    return ct_flattened, mask_flattened