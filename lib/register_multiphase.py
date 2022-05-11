import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import time

data_dir = './ml_ready_phaselabel'
outdata_dir = data_dir + '_registration'

HCC_list = os.listdir(data_dir)
HCC_list = [hcc for hcc in HCC_list if 'HCC' in hcc]
HCC_list.sort(reverse=True)

stime = time.time()

for hcc in HCC_list:
    data_list = os.listdir(os.path.join(data_dir, hcc))
    data_list = [data for data in data_list if 'ct' in data]
    data_list.sort()
    os.makedirs(os.path.join(outdata_dir, hcc), exist_ok=True)

    for data in data_list:
        if not os.path.isfile(os.path.join(outdata_dir, hcc, data)):
            print("Processing: ", os.path.join(data_dir, hcc, data))
            npy_data = np.load(os.path.join(data_dir, hcc, data))
            total_out_data = []
            for i in range(3):
                fixed_img = sitk.GetImageFromArray(npy_data[2, i])
                out_data = []
                for j in range(4):
                    if j != 2:
                        moving_img = sitk.GetImageFromArray(npy_data[j, i])
                        elastixImageFilter = sitk.ElastixImageFilter()
                        elastixImageFilter.SetFixedImage(fixed_img)
                        elastixImageFilter.SetMovingImage(moving_img)

                        parameterMapVector = sitk.VectorOfParameterMap()
                        parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
                        parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
                        elastixImageFilter.SetParameterMap(parameterMapVector)

                        elastixImageFilter.Execute()
                        #     sitk.WriteImage(elastixImageFilter.GetResultImage())
                        resultImg = elastixImageFilter.GetResultImage()
                        result_img = sitk.GetArrayFromImage(resultImg)
                        out_data.append(result_img)
                    else:
                        out_data.append(npy_data[2, i])
                total_out_data.append(np.stack(out_data, axis=0))
            final_out = np.stack(total_out_data, axis=1)
            np.save(os.path.join(outdata_dir, hcc, data), final_out)
            with open('registration2.txt', 'a') as f:
                f.write('{}\n'.format(os.path.join(hcc, data)))
            print(time.time() - stime)