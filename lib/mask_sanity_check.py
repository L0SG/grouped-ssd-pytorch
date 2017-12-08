import utils
# this code is for checking data sanity of mask data
# debug purpose only
print('reading dicom and mask data...')

list_ct, list_mask = utils.load_liver_seg_dataset(data_path='/home/tkdrlf9202/Datasets/liver_sanitycheck',
                                                  num_data_to_load=None)