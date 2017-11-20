from PIL import Image
import os
import numpy as np

data_path = '/home/tkdrlf9202/Datasets/lesion_sample'

img_bmp = Image.open(os.path.join(data_path, 'A0000_ano.bmp'))
img_jpg = Image.open(os.path.join(data_path, 'A0000_ano.jpg'))
img_png = Image.open(os.path.join(data_path, 'A0000_ano.png'))
img_tiff = Image.open(os.path.join(data_path, 'A0000_ano.tiff'))

array_bmp = np.array(img_bmp).transpose(2, 0, 1)
array_jpg = np.array(img_jpg).transpose(2, 0, 1)
array_png = np.array(img_png).transpose(2, 0, 1)
array_tiff = np.array(img_tiff).transpose(2, 0, 1)

array_bmp_r = array_bmp[0]
print('none')