import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import time

data_dir = './ml_ready_phaselabel'
outdata_dir = data_dir + '_align'

HCC_list = os.listdir(data_dir)
HCC_list = [hcc for hcc in HCC_list if 'HCC' in hcc]
HCC_list.sort(reverse=True)

stime= time.time()

for hcc in HCC_list:
    data_list = os.listdir(os.path.join(data_dir, hcc))
    data_list = [data for data in data_list if 'ct' in data]
    data_list.sort()
    os.makedirs(os.path.join(outdata_dir, hcc), exist_ok=True)
    
    for data in data_list:
        if not os.path.isfile(os.path.join(outdata_dir, hcc, data)):
            print("Processing: ", os.path.join(data_dir, hcc, data))
            ct_data = np.load(os.path.join(data_dir, hcc, data))
            phase = np.load(os.path.join(data_dir, hcc, data.replace('ct', 'phase')))
            mask_data = np.load(os.path.join(data_dir, hcc, data.replace('ct', 'mask')))
            total_out_data = []
            for i in range(3):
                fixed_img = sitk.GetImageFromArray(ct_data[2,i])
                out_data = []
                for j in range(4):
                    if j !=2:
                        moving_img = sitk.GetImageFromArray(ct_data[j,i])
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
                        
#                         sitk.WriteParameterFile(elastixImageFilter.GetTransformParameterMap()[0], '/tmp/param0.txt')
                  
                        if i == 1 and phase == j:
                            transformParameterMap = elastixImageFilter.GetTransformParameterMap()
                            transformixImageFilter = sitk.TransformixImageFilter()
                            transformixImageFilter.SetTransformParameterMap(transformParameterMap)
                            transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(mask_data))
                            transformixImageFilter.Execute()
                            resultMask = transformixImageFilter.GetResultImage()
                            result_mask = sitk.GetArrayFromImage(resultMask)
                            np.save(os.path.join(outdata_dir, hcc, data.replace('ct', 'mask')), result_mask)
                    else:
                        out_data.append(ct_data[2,i])
                        if i == 1 and phase == j:
                            np.save(os.path.join(outdata_dir, hcc, data.replace('ct', 'mask')), mask_data)
                total_out_data.append(np.stack(out_data, axis=0))
            final_out = np.stack(total_out_data, axis=1)
            np.save(os.path.join(outdata_dir, hcc, data), final_out)
            with open('align.txt', 'a') as f:
                f.write('{}\n'.format(os.path.join(hcc, data)))
            print(time.time()-stime)
            
