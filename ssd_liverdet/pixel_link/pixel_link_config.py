
version = "4s"
# epoch = 60000
# all_trains = 1000
dilation = True
use_crop = False
use_rotate = True
# iterations = 10
gpu = True
multi_gpu = False # only useful when gpu=True
pixel_weight = 2
link_weight = 1

r_mean = 49.
g_mean = 49.
b_mean = 49.

image_height = 300
image_width = 300
image_channel = 3

link_weight = 1
pixel_weight = 2
neg_pos_ratio = 3 # parameter r in paper
min_area = 3
min_height = 1

decode_method = "DECODE_METHOD_join"
pixel_conf_threshold = 0.2
link_conf_threshold = 0.8

# model arch-related
vgg_groups = 4
feature_scale = 1




# train_images_dir = "/home/tkdrlf9202/Datasets/ICDAR_2015/train_images/"
# train_labels_dir = "/home/tkdrlf9202/Datasets/ICDAR_2015/train_gt/"
# saving_model_dir = "/media/hdd/tkdrlf9202/Experiments/gssd_baseline"