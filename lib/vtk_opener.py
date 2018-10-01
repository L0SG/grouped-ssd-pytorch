import os
import numpy as np
import scipy.misc
import vtk
from vtk import vtkStructuredPointsReader
from vtk.util import numpy_support as VN
from vtk.util.numpy_support import vtk_to_numpy

#%%
# load a vtk file as input
reader = vtk.vtkPolyDataReader()
reader.SetFileName("/home/tkdrlf9202/Datasets/snuh_HCC_sample_1807/MEDIP/HCC_1104.vtk")
reader.ReadAllScalarsOn()
reader.ReadAllVectorsOn()
reader.Update()

#%%
polydata = reader.GetOutput()
points = polydata.GetPoints()
array = points.GetData()
numpy_nodes = vtk_to_numpy(array)

#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(numpy_nodes[:, 0], numpy_nodes[:, 1], numpy_nodes[:, 2])

plt.show()
