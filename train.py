from lib import utils
from lib import unet
import os.path
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torchvision
import h5py

# set GPU ID to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# path for preprocessed data
preprocessed_data_path = '/home/vision/tkdrlf9202/Datasets/liver_preprocessed/liver_data.h5'

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
    list_ct, list_mask = utils.load_liver_seg_dataset(data_path='/home/vision/tkdrlf9202/Datasets/liver',
                                                      num_data_to_load=None)
    # preprocess the dataset
    print('preprocessing dataset...')
    list_ct_preprocessed, list_mask_preprocessed = utils.preprocess_liver_dataset(list_ct, list_mask)

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

# wrap the data to pytorch tensordataset and construct dataloader
print('constructing dataloader...')
liver_tensor_dataset = TensorDataset(data_tensor=torch.FloatTensor(ct_flattened),
                                     target_tensor=torch.LongTensor(mask_flattened.astype(np.int64)))
liver_dataloader = DataLoader(dataset=liver_tensor_dataset, batch_size=8,
                              shuffle=True, drop_last=True)


# define unet for binary pixel-wise segmentation
print('loading the model...')
#model_unet = unet.unet(feature_scale=1, n_classes=2, is_deconv=True, in_channels=1, is_batchnorm=True).cuda()
model_unet = torch.nn.DataParallel(
  unet.unet(feature_scale=1, n_classes=2, is_deconv=True, in_channels=1, is_batchnorm=True).cuda())

# define the optimizer
# optimizer = torch.optim.Adam(params=model_unet.parameters(), lr=1e-5)
# same hyper params with cascaded FCN
optimizer = torch.optim.SGD(params=model_unet.parameters(), lr=1e-4, momentum=0.8,
                            weight_decay=0.0005, nesterov=True)

# train the model
epochs = 1000
for epoch in range(epochs):
    if not os.path.exists('train_samples'):
        os.makedirs('train_samples')
    samples_save_path = os.path.join('train_samples', 'epoch_'+str(epoch))
    if not os.path.exists(samples_save_path):
        os.makedirs(samples_save_path)
    for idx, (inputs, targets) in enumerate(liver_dataloader):
        # calculate class weight
        # currently hard-coding, needs to be more generic
        num_pixels_background = torch.numel(targets[targets == 0])
        num_pixels_foreground = torch.numel(targets[targets == 1])
        class_weight_background = float(num_pixels_foreground) / (num_pixels_background + num_pixels_background)
        class_weight_foreground = 1 - class_weight_background
        class_weight = torch.FloatTensor([class_weight_background, class_weight_foreground]).cuda()

        # wrap inputs and targets to variables
        inputs = Variable(inputs).cuda()
        targets = Variable(targets).cuda()

        # squeeze the channel dim (which is 1), to match requirement of loss function
        targets = torch.squeeze(targets, dim=1)

        # DEBUG CODE: random tensors instead of real inputs
        # inputs = Variable(torch.randn(4, 1, 572, 572).type(torch.FloatTensor)).cuda()
        # targets = Variable(torch.LongTensor(4, 388, 388).random_(0, 1)).cuda()

        # calculate softmax output from unet
        # currently softmax layer is embedded in unet class, but it's subject to change
        softmax = model_unet(inputs)

        # calculate loss and update params
        loss_nll2d = torch.nn.NLLLoss2d(weight=class_weight)
        loss = loss_nll2d(torch.log(softmax), targets)
        model_unet.zero_grad()
        loss.backward()
        optimizer.step()

        # print current epoch, step and loss
        if (idx + 1) % 1 == 0:
            print('epoch ' + str(epoch) + ' step ' + str(idx+1) + ' loss ' + str(loss.data[0]))
        # save inputs, targets and sigmoid outputs to image
        if (idx + 1) % 50 == 0:
            torchvision.utils.save_image(inputs.data,
                                         os.path.join(samples_save_path, 'input_'+str(idx)+'.jpg'))
            torchvision.utils.save_image(torch.unsqueeze(targets.data, dim=1),
                                         os.path.join(samples_save_path, 'target_'+str(idx)+'.jpg'))
            # take second channel (label 1, i.e. foreground) only and unsqueeze to match color channel
            torchvision.utils.save_image(torch.unsqueeze(softmax.data[:, 1, :, :], dim=1),
                                         os.path.join(samples_save_path, 'softmax_'+str(idx)+'.jpg'))

# save the model

print ('testtest')