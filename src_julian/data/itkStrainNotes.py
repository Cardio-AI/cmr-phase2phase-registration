####itk-strain######

import itk
import numpy as np

data = stack_nii_flowfield(PATH_TO_NII_FILES=path_to_patient_folder, naming='_flow_', Ntimesteps=N_TIMESTEPS)
data = data.astype('float64')

# reorder channel
channel_z, channel_y, channel_x = (data[..., 0], data[..., 1], data[..., 2])
data[..., 0] = channel_x
data[..., 1] = channel_y
data[..., 2] = channel_z

flowfield = mvf(data, '4Dt', Z_SPACING)

# reorder array
data=np.einsum('zyxc->cxyz', flowfield.Data[0]) # 3,128,128,64
INFO(data.shape)

# data = np.random.random((3, 64, 64, 64))
DisplacementImageType = itk.Image[itk.Vector[itk.D, 3], 3]
displacement = itk.image_view_from_array(data, ttype=DisplacementImageType)
strain_filter = itk.StrainImageFilter[DisplacementImageType, itk.D, itk.D].New(displacement)
strain_filter.SetStrainForm(strain_filter.StrainFormType_GREENLAGRANGIAN) #StrainFormType_INFINITESIMAL, StrainFormType_EULERIANALMANSI
strain_filter.Update()
strain = strain_filter.GetOutput()
test = itk.GetArrayFromImage(strain)
INFO(test.shape)

# create sample matrix and displacementfield
import numpy as np
import itk

# sample_matrix = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype='float64')

sample_matrix = np.ndarray((2, 2, 2))
sample_matrix[0] = np.array([[0, 1],[0, 1]])
sample_matrix[1] = np.array([[0, 0],[1, 1]])

# sample_matrix = np.reshape(np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype='float64'), (2,2,2))
# sample_displacementfield = np.reshape(np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype='float64'), (2,2,2))

sample_displacementfield = np.ndarray((2, 2, 2))
sample_displacementfield[0] = np.array([[-1, 0],[0, 0]])
sample_displacementfield[1] = np.array([[-1, 0],[0, 0]])
Dimension = 2
DisplacementImageType = itk.Image[itk.Vector[itk.D, Dimension], Dimension]
displacement = itk.image_view_from_array(sample_displacementfield, ttype=DisplacementImageType)
strain_filter = itk.StrainImageFilter[DisplacementImageType, itk.D, itk.D].New(displacement)
strain_filter.SetStrainForm(strain_filter.StrainFormType_INFINITESIMAL) #StrainFormType_INFINITESIMAL, StrainFormType_EULERIANALMANSI, StrainFormType_GREENLAGRANGIAN
strain_filter.Update()
strain = strain_filter.GetOutput()
output_strain = itk.GetArrayFromImage(strain)

slice=20
fig, ax = plt.subplots(1, 6, figsize=(20,5), sharey=True)
for i in range(6):
    ax[i].imshow(test[slice,:,:,i], cmap='jet')
    ax[i].grid(False)
# plt.colorbar()
plt.show()