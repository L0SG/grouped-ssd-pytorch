from lib import utils
from lib import unet
from lib import datahandler
import os.path
import sys
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import torchvision
import h5py

# GPU ID to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

# hyperparameters
learning_rate = 1e-4
momentum = 0.8
weight_decay = 0.0005
batch_size = 1

# path for original & preprocessed data
preprocessed_data_path = '/home/tkdrlf9202/Datasets/liver_preprocessed/liver_data.h5'
data_path = '/media/hdd/tkdrlf9202/Datasets/liver'

# load preprocessed dataset
ct_flattened, mask_flattened = datahandler.load_liver_dataset(preprocessed_data_path, data_path)
ct_train, ct_valid, mask_train, mask_valid = train_test_split(ct_flattened, mask_flattened, test_size=0.1)

# wrap the data to pytorch tensordataset and construct dataloader
print('constructing dataloader...')
liver_tensor_dataset_train = TensorDataset(data_tensor=torch.FloatTensor(ct_train),
                                           target_tensor=torch.LongTensor(mask_train.astype(np.int64)))
liver_tensor_dataset_valid = TensorDataset(data_tensor=torch.FloatTensor(ct_valid),
                                           target_tensor=torch.LongTensor(mask_valid.astype(np.int64)))

liver_dataloader_train = DataLoader(dataset=liver_tensor_dataset_train, batch_size=batch_size,
                                    shuffle=True, drop_last=True)
liver_dataloader_valid = DataLoader(dataset=liver_tensor_dataset_valid, batch_size=batch_size,
                                    shuffle=True, drop_last=True)

# define unet for binary pixel-wise segmentation
print('loading the model...')
#model_unet = unet.unet(feature_scale=1, n_classes=2, is_deconv=True, in_channels=1, is_batchnorm=True).cuda()
model_unet = torch.nn.DataParallel(
  unet.unet(feature_scale=0.5, n_classes=2, is_deconv=True, in_channels=1, is_batchnorm=True).cuda())

# load pre-trained state if exist
# TODO: make the routine generic
pretrained_path = 'model_unet_parameters_final'
if os.path.exists(os.path.join(os.getcwd(), pretrained_path)):
    print('pretrained model detected, loading parameters...')
    model_unet.load_state_dict(torch.load(os.path.join(os.getcwd(), pretrained_path)))
    print('parameter loaded')

# define the optimizer
# optimizer = torch.optim.Adam(params=model_unet.parameters(), lr=1e-5)
# same hyper params with cascaded FCN
optimizer = torch.optim.SGD(params=model_unet.parameters(), lr=learning_rate,
                            momentum=momentum, weight_decay=weight_decay, nesterov=True)

# path for results & logs
results_path = 'results_debug_'+str(learning_rate)+'_'+str(momentum)+'_'+str(weight_decay)
if not os.path.exists(results_path):
    os.makedirs(results_path)
# make log file
logger_train = open(os.path.join(results_path, 'train_log.txt'), 'w')
logger_valid = open(os.path.join(results_path, 'valid_log.txt'), 'w')
# path for trained model parameters
model_param_path = 'model_unet_parameters'

# train and validate the model
epochs = 100
for epoch in range(epochs):
    samples_save_path = os.path.join(results_path, 'epoch_'+str(epoch))
    if not os.path.exists(samples_save_path):
        os.makedirs(samples_save_path)

    # training phase
    running_loss = 0.
    for idx, (inputs, targets) in enumerate(liver_dataloader_train):
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
        # add step loss to running loss
        running_loss += loss.data[0]

        # print current epoch, step and avg.loss ; then save to logger
        if (idx + 1) % 10 == 0:
            running_loss /= 10
            logging_data = 'epoch ' + str(epoch) + ' step ' + str(idx+1) + ' loss ' + str(running_loss)
            print(logging_data)
            logger_train.write(logging_data + '\n')
            logger_train.flush()
            running_loss = 0.

        # save inputs, targets and softmax outputs to image
        if (idx + 1) % 500 == 0:
            torchvision.utils.save_image(inputs.data,
                                         os.path.join(samples_save_path, 'input_'+str(idx)+'.jpg'))
            torchvision.utils.save_image(torch.unsqueeze(targets.data, dim=1),
                                         os.path.join(samples_save_path, 'target_'+str(idx)+'.jpg'))
            # take second channel (label 1, i.e. foreground) only and unsqueeze to match color channel
            torchvision.utils.save_image(torch.unsqueeze(softmax.data[:, 1, :, :], dim=1),
                                         os.path.join(samples_save_path, 'softmax_'+str(idx)+'.jpg'))

    # delete unnecessary graphs to free up vram
    del inputs, targets, softmax, loss

    # validation phase
    print('epoch finished, running validation phase...')
    valid_loss = 0.
    for idx_valid, (inputs, targets) in enumerate(liver_dataloader_valid):
        # same as training phase
        num_pixels_background = torch.numel(targets[targets == 0])
        num_pixels_foreground = torch.numel(targets[targets == 1])
        class_weight_background = float(num_pixels_foreground) / (num_pixels_background + num_pixels_background)
        class_weight_foreground = 1 - class_weight_background
        class_weight = torch.FloatTensor([class_weight_background, class_weight_foreground]).cuda()

        # use volatile flag to reduce memory usage (disable BP)
        inputs = Variable(inputs, volatile=True).cuda()
        targets = Variable(targets, volatile=True).cuda()

        targets = torch.squeeze(targets, dim=1)

        softmax = model_unet(inputs)

        loss_nll2d = torch.nn.NLLLoss2d(weight=class_weight)
        loss = loss_nll2d(torch.log(softmax), targets)
        # add to total validation loss
        valid_loss += loss.data[0]
    # divide with total step index
    valid_loss = valid_loss / idx_valid
    # print and log
    print('validation loss: ' + str(valid_loss))
    logger_valid.write('validation loss: ' + str(valid_loss) + '\n')
    logger_valid.flush()
    # save the model
    torch.save(model_unet.state_dict(), os.path.join(results_path, model_param_path+'_epoch_'+str(epoch)))
    print('model saved')

# save the model
torch.save(model_unet.state_dict(), os.path.join(results_path, model_param_path+'_final'))
print('model saved')
