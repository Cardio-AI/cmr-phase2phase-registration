# define logging and working directory
# from ProjectRoot import change_wd_to_project_root
# change_wd_to_project_root()

# import helper functions
import numpy as np
import pandas as pd
import SimpleITK as sitk
import logging
from logging import info as INFO
from src_julian.utils.skhelperfunctions import Console_and_file_logger

# set up logging
Console_and_file_logger('mvfviz/dmd_temp', logging.INFO)

# start
class mvf:
    def __init__(self, data, format, zspacing):
        self.Data = data
        self.Format = format
        self.Z_Spacing = zspacing
        self.Shape = data.shape
        self.Dimension = data.ndim
        self.get_specs()

    def get_specs(self):
        if self.Format == '3D' and self.Dimension == 3:
            self.nt = 1
            self.ny, self.nx, self.nc = self.Shape
        if self.Format == '3Dt' and self.Dimension == 4:
            self.nt, self.ny, self.nx, self.nc = self.Shape
        if self.Format == '4D' and self.Dimension == 4:
            self.nt = 1
            self.nz, self.ny, self.nx, self.nc = self.Shape
        if self.Format == '4Dt' and self.Dimension == 5:
            self.nt, self.nz, self.ny, self.nx, self.nc = self.Shape

    def save_as_nii(self, filename, PATH_TO_OUTPUT_FOLDER):
        if self.Format == '3D' and self.Dimension == 3:
            ff_rearranged = np.einsum('yxc->cxy', self.Data)
            INFO('shape of ff after rearrangement ready for nii export (c, x, y): {}'.format(ff_rearranged.shape))
        elif self.Format == '3Dt' and self.Dimension == 4:
            ff_rearranged = np.einsum('tyxc->cxyt', self.Data)
            INFO('shape of ff after rearrangement ready for nii export (c, x, y, t): {}'.format(ff_rearranged.shape))
        elif self.Format == '4D' and self.Dimension == 4:
            ff_rearranged = np.einsum('zyxc->cxyz', self.Data)
            INFO('shape of ff after rearrangement ready for nii export (c, x, y, z): {}'.format(ff_rearranged.shape))
        elif self.Format == '4Dt' and self.Dimension == 5:
            ff_rearranged = np.einsum('tzyxc->cxyzt', self.Data)
            INFO('shape of ff after rearrangement ready for nii export (c, x, y, z, t): {}'.format(ff_rearranged.shape))
            INFO('WARNING: 4Dt array display is not possible in MITK Diffusion')

        back_ff2img = sitk.GetImageFromArray(ff_rearranged, isVector=False)  # isVector=False important for adequate writing of sitk
        sitk.WriteImage(back_ff2img, PATH_TO_OUTPUT_FOLDER + filename + '.nii')
        INFO('ff {} has been successfully saved.'.format(filename))

    def save_as_nrrd(self, filename, PATH_TO_OUTPUT_FOLDER):
        sitk_images = []
        if self.nt > 1:
            for time in self.Data:
                sitk_images.append(sitk.GetImageFromArray(time))
            sitk_image = sitk.JoinSeries(sitk_images)
            sitk.WriteImage(sitk_image, PATH_TO_OUTPUT_FOLDER + filename + '.nrrd')
            INFO('File saved as .nrrd')
        elif self.nt == 1:
            INFO('save_as_nrrd for static flowfield not yet implemented.')


    def save_as_csv(self, filename, PATH_TO_OUTPUT_FOLDER):
        for time in range(self.nt): # for every timestep
            if self.nt == 1: # one timestep only
                phase = self.Data
            elif self.nt > 1: # more than one timestep
                phase = self.Data[time] # will be used if there is more than one timestep

            starting_value = 0
            x_ = np.linspace(starting_value, self.nx, self.nx)
            y_ = np.linspace(starting_value, self.ny, self.ny)
            z_ = np.linspace(starting_value, self.nz, self.nz)
            X, Y, Z = np.meshgrid(x_, y_, z_)
            X = X.T
            Y = Y.T
            Z = Z.T
            x = X.ravel()
            y = Y.ravel()
            z = Z.ravel()
            z = z * self.Z_Spacing

            if self.nt == 1:
                z_direction = self.Data[..., 0]
                y_direction = self.Data[..., 1]
                x_direction = self.Data[..., 2]
            elif self.nt > 1:
                z_direction = self.Data[time,:,:,:,0]
                y_direction = self.Data[time,:,:,:,1]
                x_direction = self.Data[time,:,:,:,2]

            Fz = z_direction.ravel()
            Fy = y_direction.ravel()
            Fx = x_direction.ravel()
            df = pd.DataFrame({"x": x, "y": y, "z": z, "Fx": Fx, "Fy": Fy, "Fz": Fz})
            df.to_csv(PATH_TO_OUTPUT_FOLDER + filename + ".csv." + str(time), index=False)
            # INFO('csv timestep ' + str(time) + ' successfully saved.')

    def plot_Grid2D_MV2Dor3D(self, slice, N):
        # return xx, yy, Fx, Fy
        # this method quickly visualizes motion vectors of a 2D slice from a 3D volume
        # MVF vector ordering has to be z,y,x,c
        # N = slicing, take every Nth value
        # a function that handles 2D grids such as 224x224 and plots motion vectors 2D on the grid in the xy plane.
        # it is not yet stated how z values of 3D MV should be visualized here.

        # EXAMPLE CALL:
        # t = 2
        # slice = 5
        # N = 5
        # whole has dimensions tzyxc with c.ndim=3 zyx
        # test = mvf(data=whole[t, ...], format='4D', zspacing=1)
        # xx, yy, Fx, Fy = test.plot_Grid2D_MV2Dor3D(slice=slice, N=N)
        # fig, ax = plt.subplots()
        # plt.quiver(xx, yy, Fx, Fy, units='xy', angles='xy', scale=1) #-yy for img orientation origin upper left
        # ax.set_title('MVF plot')
        # ax.set_aspect('equal')
        # plt.show()

        # voxelmorph tutorial visualization: tutorial.voxelmorph.net
        # imports
        # import os, sys
        # third party imports
        # import numpy as np
        # import tensorflow as tf
        # assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
        # import voxelmorph as vxm
        # import neurite as ne
        # N = 1
        # flow = ff[t, z, ::N, ::N, 1:3] 1:3 means x and y coordinates for vector plotting
        # ne.plot.flow([flow], width=5)

        if self.Shape[-1] == 2:  # MVs are 2D
            # if the motion vectors input are in 2D coordinates, we are filling with 0 z values
            print("MVs: 2D")
            print("Vectors are now filled with zeros as z-values")
            b = np.zeros((self.nz, self.ny, self.nx, self.Shape[-1] + 1)) # increase MVs dimensions by 1, resp. z-values
            b[..., 1:] = self.Data
            self.Data = b
        elif self.Shape[-1] == 3:  # MVs are 3D
            print("MVs: 3D")

        # calculation of grid coordinates begins
        print("Calculating now.")

        zdir = self.Data[slice, :, :, 0]  # motion vectors z directions
        ydir = self.Data[slice, :, :, 1]  # motion vectors y directions
        xdir = self.Data[slice, :, :, 2]  # motion vectors x directions

        # slicing the field, take every Nth value
        Fz = zdir[::N, ::N]
        Fy = ydir[::N, ::N]
        Fx = xdir[::N, ::N]
        nrows, ncols = Fx.shape

        starting_value = 0
        zstart = starting_value
        ystart = starting_value
        xstart = starting_value

        # these linspace values are later used for meshgrid
        zlin = np.linspace(zstart, self.nz - 1, self.nz)
        ylin = np.linspace(ystart, self.ny - 1, nrows)
        xlin = np.linspace(xstart, self.nx - 1, ncols)

        # calculating the meshgrid which equals the vectors origins
        xx, yy = np.meshgrid(xlin, ylin, indexing='xy')
        zz = np.meshgrid(zlin, self.nz, indexing='xy')

        # -----------------------------------#
        # here we have to state what should happen with the z values of the MVs on a 2D grid!
        # -----------------------------------#

        return xx, yy, Fx, Fy

    def switch_channel(self):
        '''
        switches the channel entries from zyx to xyz or the other way round
        for this to be applied, the channel has to be the last dimension
        --> from zyx to xyz means from 012 to 210
        '''
        return np.stack([self.Data[..., 2],
                         self.Data[..., 1],
                         self.Data[..., 0]], axis=-1)

    def compose_justadding(self):
        '''
        implemented for 5 timesteps !
        just adds the timesteps of a flowfield on top of each other
        '''
        # return self.Data[0:5].sum(axis=0)

        return np.stack([self.Data[0],
                        self.Data[0] + self.Data[1],
                        self.Data[0] + self.Data[1] + self.Data[2],
                        self.Data[0] + self.Data[1] + self.Data[2] + self.Data[3],
                        self.Data[0] + self.Data[1] + self.Data[2] + self.Data[3] + self.Data[4]],
                        axis=0)

    def compose_sitk(self, Z_SLICES, method):
        '''
        flowfield composition with sitk algorithms
        input: tzyxc with c=zyx
        output: same
        '''

        # get a list with n timesteps of displacement fields transforms.
        # needed to compose the timesteps afterwards.
        t = []
        for timestep in range(self.nt):
            arr = self.Data[timestep, Z_SLICES, ...].astype('float64')
            img = sitk.GetImageFromArray(arr=arr, isVector=True)
            img.SetOrigin(origin=(0, 0, 0))
            img.SetSpacing(spacing=(1, 1, 1))
            t.append(sitk.DisplacementFieldTransform(img))

        # "The transforms are composed in reverse order with the back being applied first"
        # https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1CompositeTransform.html
        if method == 'forward':
            transform0 = sitk.CompositeTransform([t[0]])
            transform1 = sitk.CompositeTransform([t[0], t[1]])
            transform2 = sitk.CompositeTransform([t[0], t[1], t[2]])
            transform3 = sitk.CompositeTransform([t[0], t[1], t[2], t[3]])
            transform4 = sitk.CompositeTransform([t[0], t[1], t[2], t[3], t[4]])
        if method== 'reversed':
            transform0 = sitk.CompositeTransform([t[0]])
            transform1 = sitk.CompositeTransform([t[1], t[0]])
            transform2 = sitk.CompositeTransform([t[2], t[1], t[0]])
            transform3 = sitk.CompositeTransform([t[3], t[2], t[1], t[0]])
            transform4 = sitk.CompositeTransform([t[4], t[3], t[2], t[1], t[0]])

        transforms = np.stack([transform0, transform1, transform2, transform3, transform4])

        ff_comp_itk = np.ndarray((self.nt, len(Z_SLICES), self.ny, self.nx, 3))
        df = sitk.TransformToDisplacementFieldFilter()
        df.SetReferenceImage(refImage=sitk.GetImageFromArray(arr=self.Data[0, Z_SLICES, ...], isVector=True))
        for timestep in range(self.nt):
            ff_comp_itk[timestep, ...] = sitk.GetArrayFromImage(df.Execute(transforms[timestep]))

        return ff_comp_itk

    def compose_myimplementation(self):
        '''
        takes the flowfield and sums up the flowfield vectors for every time step
        the resulting flowfield then contains:
        0: ED to MS flowfield
        1: ED to ES flowfield
        2: ED to PF flowfield
        3: ED to MD flowfield
        4: ED to ED flowfield
        ####
        CAVE: not catched at the moment are values if the arrows point outside the image, then the indices get exceeded
        ####
        '''

        # inits
        target_coords = np.zeros((self.Shape))
        sum_up_ff = np.zeros((self.Shape))

        # loop over every point, over the time for each point
        for z in range(self.nz):
            for y in range(self.ny):
                for x in range(self.nx):
                    for t in range(self.nt):
                        # get input coordinates
                        if t == 0: # ED case, first iterations. here we dont yet have target values of previous step
                            input_coord_z = z
                            input_coord_y = y
                            input_coord_x = x
                        elif t > 0:
                            input_coord_z = target_coords[t-1,z,y,x,0]
                            input_coord_y = target_coords[t-1,z,y,x,1]
                            input_coord_x = target_coords[t-1,z,y,x,2]

                        # get idxs out of exact coordinate values
                        input_idx_z = np.round(input_coord_z, decimals=0).astype(int)
                        input_idx_y = np.round(input_coord_y, decimals=0).astype(int)
                        input_idx_x = np.round(input_coord_x, decimals=0).astype(int)

                        # check if the idxs are still within the image; if not, take the edge value
                        if input_idx_z < 0: input_idx_z = 0
                        elif input_idx_z > self.nz-1: input_idx_z = self.nz-1
                        if input_idx_y < 0: input_idx_y = 0
                        elif input_idx_y > self.ny-1: input_idx_y = self.ny-1
                        if input_idx_x < 0: input_idx_x = 0
                        elif input_idx_x > self.nx-1: input_idx_x = self.nx-1


                        # get flowfield values at those idxs
                        ff_z = self.Data[t, input_idx_z, input_idx_y, input_idx_x, 0]
                        ff_y = self.Data[t, input_idx_z, input_idx_y, input_idx_x, 1]
                        ff_x = self.Data[t, input_idx_z, input_idx_y, input_idx_x, 2]

                        # accumulate ff entries
                        if t == 0: # ED case, first iterations. here we dont yet have previous ff values
                            sum_up_ff[t, z, y, x, 0] = ff_z
                            sum_up_ff[t, z, y, x, 1] = ff_y
                            sum_up_ff[t, z, y, x, 2] = ff_x
                        elif t > 0:
                            sum_up_ff[t, z, y, x, 0] = sum_up_ff[t-1, z, y, x, 0] + ff_z
                            sum_up_ff[t, z, y, x, 1] = sum_up_ff[t-1, z, y, x, 1] + ff_y
                            sum_up_ff[t, z, y, x, 2] = sum_up_ff[t-1, z, y, x, 2] + ff_x

                        # apply transform, exact calculation steps
                        target_coord_z = input_coord_z + ff_z
                        target_coord_y = input_coord_y + ff_y
                        target_coord_x = input_coord_x + ff_x

                        # save coordinates after flowfield application
                        # the idx coordinates dont have to be saved because they can be
                        # recalculated at any time out of the exact positions
                        # these values will be needed as the new input coordinates for the next iteration
                        target_coords[t, z, y, x, 0] = target_coord_z
                        target_coords[t, z, y, x, 1] = target_coord_y
                        target_coords[t, z, y, x, 2] = target_coord_x

            INFO('trajectories slice ' + str(z+1) + ' out of ' + str(self.nz) + ' finished')

        return sum_up_ff, target_coords



class volume(mvf):
    def get_specs(self):
        if self.Format == '3D' and self.Dimension == 3:
            self.nt = 1
            self.ny, self.nx, self.nv = self.Shape
        if self.Format == '3Dt' and self.Dimension == 4:
            self.nt, self.ny, self.nx, self.nv = self.Shape
        if self.Format == '4D' and self.Dimension == 4:
            self.nt = 1
            self.nz, self.ny, self.nx, self.nv = self.Shape
        if self.Format == '4Dt' and self.Dimension == 5:
            self.nt, self.nz, self.ny, self.nx, self.nv = self.Shape


    def save_as_csv(self, filename, PATH_TO_OUTPUT_FOLDER):
        # now we are saving data for every timestep
        for time in range(self.nt):
            if self.nt == 1: # one timestep only
                phase = self.Data
            elif self.nt > 1:
                phase = self.Data[time]  # will be used if there is more than one timestep

            starting_value = 0
            x_ = np.linspace(starting_value, self.nx, self.nx)
            y_ = np.linspace(starting_value, self.ny, self.ny)
            z_ = np.linspace(starting_value, self.nz, self.nz)
            X, Y, Z = np.meshgrid(x_, y_, z_)
            X = X.T
            Y = Y.T
            Z = Z.T
            x = X.ravel()
            y = Y.ravel()
            z = Z.ravel()
            z = np.multiply(z, self.Z_Spacing)

            if self.nt == 1:
                values = self.Data[..., 0]
            elif self.nt > 1:
                values = self.Data[time,:,:,:,0]

            # values = values.T if data is not "einsummed" to have the order z,y,x,c
            values = values.ravel()
            df = pd.DataFrame({"x": x, "y": y, "z": z, "value": values})
            df.to_csv(PATH_TO_OUTPUT_FOLDER + filename + ".csv." + str(time), index=False)
            INFO('csv timestep ' + str(time) + ' successfully saved.')

    def get_voxelcount(self, idx_to_be_counted, N_TIMESTEPS):
        '''
        takes the segmentation volume, receives the indexes of segmentations
        then counts for all timesteps desired the amount of the index needed
        i.e. takes 30,15,256,56,1 segmentation array
        returns array of shape 5,1 where 5 timesteps and 1 counter value
        the counter value states how many voxels are with the desired index at the timesteps
        i.e. 5000 voxels at ED, 2000 at ES etc.
        '''

        t=0 # first phase

        # check, in which timesteps we do have a segmentation
        seg_array_timesteps = self.get_segmentationarray('timestepwise')
        seg_array_slicewise = self.get_segmentationarray('slicewise')
        seg_array_pixelwise = self.get_segmentationarray('pixelwise')

        # in this array, we save the total number of segmented pixels for every timestep
        # counter_array[0] equals the number of MS pixels i.e.
        counter_array = np.zeros((N_TIMESTEPS, 1))

        # for segmentation timesteps, count segmented pixels
        # identify timestep where we have something
        for idx, num in enumerate(seg_array_timesteps):
            if num:
                # identify slices at that timestep where we have something
                for idx2, num2 in enumerate(seg_array_slicewise[idx]):
                    if num2:
                        # take the image which we want to count
                        # we could also take the whole volume but this makes the code better for proofreading
                        img_tobecounted = self.Data[idx, idx2, :, :, 0]

                        # flatten before counting
                        img_flattened = img_tobecounted.flatten()

                        # count the index
                        count_arr = np.bincount(img_flattened)

                        # at the boundaries, there might be no desired segmentation available
                        if len(count_arr) >= idx_to_be_counted:
                            # add the current value to the counter array
                            counter_array[t] = counter_array[t] + count_arr[idx_to_be_counted]
                t = t + 1 # go to the next timestep

                # account for errors
                if t > 5:
                    print('ERROR: found too many segmentation timesteps')
        return counter_array


    def get_segmentationarray(self, resolution):
        '''
        takes a segmentation volume of format i.e. 30,15,128,128,1
        returns a boolarray of format i.e. 30,15,1 which states in which slices we have segmentations
        '''

        if self.Format == '4Dt':
            if resolution == 'timestepwise':
                # create empty array full of False
                boolarray = np.full((self.nt, 1), False)
                for t in range(self.nt):
                    if np.any(self.Data[t, :, :, :, 0]):
                        boolarray[t, 0] = True

            if resolution == 'slicewise':
                # create empty array full of False
                boolarray = np.full((self.nt, self.nz, 1), False)

                for t in range(self.nt):
                    for z in range(self.nz):
                        if np.any(self.Data[t, z, :, :, 0]):
                            boolarray[t, z, 0] = True

            elif resolution == 'pixelwise':
                # create empty array full of False
                boolarray = np.full((self.nt, self.nz, self.ny, self.nx, 1), False)

                # check all pixels for segmentations
                boolarray = (self.Data != 0)


        return boolarray

    def get_Farneback_flowfield_of_slice(self, slice, reference, parameters):
        # set array[t] for wandering and array[0] for static ED reference

        import numpy as np
        import cv2 as cv
        import matplotlib.pyplot as plt

        # we will only calculate the flowfield of one slice here, so we select it
        # we copy to not manipulate the original data
        data = np.copy(self.Data[:, slice, ...])

        # at the beginning we do have a volume which contains arbitrary grayvalues
        # convert arbitrary grayscale values of array into grayscale 0-255
        # array = array/(array.max()-array.min())*255
        # this minmax normalization is also already available as an implementation
        # if we have a phantom dataset here, we have values between 0 and 1 usually which will be converted
        array = cv.normalize(data, None, 0, 255, cv.NORM_MINMAX)

        # inits
        flow = np.ndarray((self.nt, self.ny, self.nx, 2))
        hsv = np.ndarray((self.nt, self.ny, self.nx, 3), dtype='uint8')  # hue, saturation, color
        bgr_array = np.ndarray((self.nt, self.ny, self.nx, 3), dtype='uint8')

        # set the hsv saturation values to 255 maximum saturation
        hsv[..., 1] = 255

        # iterations
        for t in range(self.nt):
            # define what is the current image
            if reference=='ED': curr = array[0]
            elif reference=='dynamic': curr = array[t]

            # define what is the next image
            # if we are at the last position, that is MD, there will not be a next stack. thus, we take the first again.
            # this means that the last flowfield will be from MD to ED again
            if t==self.nt-1: next = array[0]
            else: next = array[t + 1]

            # convert the images to uint8
            curr = curr.astype('uint8')
            next = next.astype('uint8')

            # defaults from page: pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            # my testing for phantom: pyr_scale=0.5,levels=5,winsize=3, iterations=1,poly_n=7,poly_sigma=1.5,flags=None
            flow[t] = cv.calcOpticalFlowFarneback(prev=curr, next=next, flow=None,
                                                  pyr_scale=parameters[0], levels=parameters[1], winsize=parameters[2],
                                                  iterations=parameters[3], poly_n=parameters[4], poly_sigma=parameters[5], flags=parameters[6])

            mag, ang = cv.cartToPolar(flow[t, ..., 0], flow[t, ..., 1])

            # set the hue values
            hsv[t, ..., 0] = ang * 180 / np.pi / 2

            # set the color values
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

            # hsv is float64 here
            # cvtColor HSV2BGR needs at max float32
            # hsv = hsv.astype('uint8')
            # hsv array is above defined as uint8

            bgr = cv.cvtColor(hsv[t], cv.COLOR_HSV2BGR)
            cv.imshow('frame2', bgr)
            k = cv.waitKey(100) & 0xff

            # save bgr values
            # bgr values are uint8, bgr_array is float64
            bgr_array[t, ...] = bgr

            # INFO('Hi')

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(bgr,alpha=1)
            # plt.imshow(curr, alpha=1)
            # plt.imshow(next, alpha=1)

            INFO('Farneback timestep ' + str(t+1) + ' / ' + str(self.nt))

        return flow, bgr_array
    
    