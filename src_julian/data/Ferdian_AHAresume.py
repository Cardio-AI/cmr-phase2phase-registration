#####################################################################################
# aha tests
maskbased_distance_cube = distance_cube
ff_distance_cube = np.roll(ff_l_cube, 1, axis=1)
maskbased_distances_aha = sort_radians_into_aha(evaluation_angles, maskbased_distance_cube, N_EVALUATION_RADIAL_AHA)
ff_distances_aha = sort_radians_into_aha(evaluation_angles, ff_distance_cube, N_EVALUATION_RADIAL_AHA)

# calculate difference cubes aha based
# distances are 6,5,14,10 as aha,t,z,value

maskbased_distances_aha_mean = maskbased_distances_aha.mean(axis=-1)
ff_distances_aha_mean = ff_distances_aha.mean(axis=-1)

l0 = maskbased_distances_aha_mean[:,0,:]
l0 = l0[:,np.newaxis,:]
l0 = np.repeat(l0, 5, axis=1)

# calculate differences and strains
maskbased_differences_aha_mean = maskbased_distances_aha_mean-l0
ff_differences_aha_mean = ff_distances_aha_mean-l0
maskbased_strain_aha_mean = maskbased_differences_aha_mean / l0 * 100
ff_strain_aha_mean = ff_differences_aha_mean / l0 * 100

# mean over all Z slices
maskbased_distances_aha_mean = maskbased_distances_aha_mean.mean(axis=-1)
maskbased_differences_aha_mean = maskbased_differences_aha_mean.mean(axis=-1)
maskbased_strain_aha_mean = maskbased_strain_aha_mean.mean(axis=-1)
ff_distances_aha_mean = ff_distances_aha_mean.mean(axis=-1)
ff_differences_aha_mean = ff_differences_aha_mean.mean(axis=-1)
ff_strain_aha_mean = ff_strain_aha_mean.mean(axis=-1)


plot_2x3distdiffstrain_AHA(maskbased_distances_aha_mean,
                       maskbased_differences_aha_mean,
                       maskbased_strain_aha_mean,
                       ff_distances_aha_mean,
                       ff_differences_aha_mean,
                       ff_strain_aha_mean,
                       N_MIDCAVITY_AHASEGMENTS, PHASE_LABELS, AHA_LABELS)




####################################################################################################################
########################################automatic reporting##############################################################

# aha coding
# aha segments
aha11_distance_array = []
aha10_distance_array = []
aha9_distance_array = []
aha8_distance_array = []
aha7_distance_array = []
aha12_distance_array = []

# get the distance values with respect to the radial angle with high resolution
# we are travelling over all evaluated angles and check to which segment these values belong
# then we assign these values into the corresponding array
for idx, angle in enumerate(evaluation_angles):
    if -np.pi/3 < angle <= 0: aha11_distance_array.append(distance_cube_mean[:, idx]) # AHA11
    if -2*np.pi/3 < angle <= -np.pi/3: aha10_distance_array.append(distance_cube_mean[:, idx]) # AHA10
    if -3*np.pi/3 < angle <= -2*np.pi/3: aha9_distance_array.append(distance_cube_mean[:, idx]) # AHA9
    if 2*np.pi/3 < angle <= 3*np.pi/3: aha8_distance_array.append(distance_cube_mean[:, idx]) # AHA8
    if np.pi/3 < angle <= 2*np.pi/3: aha7_distance_array.append(distance_cube_mean[:, idx]) # AHA7
    if 0 < angle <= np.pi/3: aha12_distance_array.append(distance_cube_mean[:, idx]) # AHA12

    # if 0 <= angle < np.pi/3: aha12_distance_array.append(distance_cube_mean[:, idx]) # AHA12
    # if np.pi/3   <= angle < 2*np.pi/3:  aha7_distance_array.append(distance_cube_mean[:, idx]) # AHA7
    # if 2*np.pi/3 <= angle < 3*np.pi/3:  aha8_distance_array.append(distance_cube_mean[:, idx]) # AHA8
    # if 3*np.pi/3 <= angle < 4*np.pi/3:  aha9_distance_array.append(distance_cube_mean[:, idx]) # AHA9
    # if 4*np.pi/3 <= angle < 5*np.pi/3:  aha10_distance_array.append(distance_cube_mean[:, idx]) # AHA10
    # if 5*np.pi/3 <= angle < 2*np.pi:    aha11_distance_array.append(distance_cube_mean[:, idx]) # AHA11

# the angle handling has been done now. this means, that the respective arrays contain the right values.
# ordering may be varied now which will not influence the values

# transpose the lists
aha11_distance_array = np.transpose(np.array(aha11_distance_array))
aha10_distance_array = np.transpose(np.array(aha10_distance_array))
aha9_distance_array = np.transpose(np.array(aha9_distance_array))
aha8_distance_array = np.transpose(np.array(aha8_distance_array))
aha7_distance_array = np.transpose(np.array(aha7_distance_array))
aha12_distance_array = np.transpose(np.array(aha12_distance_array))

# to refactor the multiple high-resolution lists, we take every Nth value to gain a np.ndarray with equal length
# per aha segment, i.e. 10 evaluation points per segment
aha11_distance_array = aha11_distance_array[:, np.round(np.linspace(0, len(aha11_distance_array) - 1, N_EVALUATION_RADIAL_AHA)).astype(int)]
aha10_distance_array = aha10_distance_array[:, np.round(np.linspace(0, len(aha10_distance_array) - 1, N_EVALUATION_RADIAL_AHA)).astype(int)]
aha9_distance_array = aha9_distance_array[:, np.round(np.linspace(0, len(aha9_distance_array) - 1, N_EVALUATION_RADIAL_AHA)).astype(int)]
aha8_distance_array = aha8_distance_array[:, np.round(np.linspace(0, len(aha8_distance_array) - 1, N_EVALUATION_RADIAL_AHA)).astype(int)]
aha7_distance_array = aha7_distance_array[:, np.round(np.linspace(0, len(aha7_distance_array) - 1, N_EVALUATION_RADIAL_AHA)).astype(int)]
aha12_distance_array = aha12_distance_array[:, np.round(np.linspace(0, len(aha12_distance_array) - 1, N_EVALUATION_RADIAL_AHA)).astype(int)]

# stack the slim arrays
aha_distance_midcavity_stack = np.stack([aha11_distance_array, aha10_distance_array,
                                aha9_distance_array, aha8_distance_array,
                                aha7_distance_array, aha12_distance_array])

# calculate the strain_midcavity_stack
aha_strain_midcavity_stack = np.ndarray((len(aha_distance_midcavity_stack), N_TIMESTEPS, N_EVALUATION_RADIAL_AHA))
for t in range(N_TIMESTEPS):
    curr_strain = (aha_distance_midcavity_stack[:, t, :]
                   - aha_distance_midcavity_stack[:, 0, :])\
                  /aha_distance_midcavity_stack[:, 0, :]
    aha_strain_midcavity_stack[:, t] = curr_strain
######################################################################################################



# plot the aha segment distance curves
plt.figure()
labels_time = ['ED', 'MS', 'ES', 'PF', 'MD']
aha_labels = ['AHA11', 'AHA10', 'AHA9', 'AHA8', 'AHA7', 'AHA12']
for idx, num in enumerate(aha_labels):
    plot(labels_time, aha_distance_midcavity_stack.mean(axis=2)[idx], label=num)
plt.xlabel('Phase')
plt.ylabel('Mean radial myocardium thickness')
plt.title('Mean radial myocardium thickness per phase')
plt.legend()
plt.show()

# plot the aha segment strain curves
plt.figure()
labels_time = ['ED', 'MS', 'ES', 'PF', 'MD']
for idx, num in enumerate(aha_labels):
    plot(labels_time, aha_strain_midcavity_stack.mean(axis=2)[idx], label=num)
plt.xlabel('Phase')
plt.ylabel('Mean radial myocardium strain')
plt.title('Mean radial myocardium strain per phase')
plt.legend()
plt.show()