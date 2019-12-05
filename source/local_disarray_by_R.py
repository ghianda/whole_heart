# system
import os
import argparse

# general
import numpy as np

# custom codes
from custom_tool_kit import manage_path_argument, create_coord_by_iter, create_slice_coordinate, all_words_in_txt, search_value_in_txt


def extract_parameters(filename, param_names, _verb = False):
    ''' read parameters values in filename.txt
    and save it in a dictionary'''

    # read values in txt
    param_values = search_value_in_txt(filename, param_names)

    print('\n ***  Parameters : \n')
    # create dictionary of parameters
    parameters = {}
    for i, p_name in enumerate(param_names):
        parameters[p_name] = float(param_values[i])
        if _verb:
            print(' - {} : {}'.format(p_name, param_values[i]))
    if _verb: print('\n \n')
    return parameters


def estimate_local_disarry(R, parameters, ev_index=2, _verb=True, _verb_deep=False):
    res_xy = parameters['px_size_xy']
    res_z = parameters['px_size_z']
    resolution_factor = res_z / res_xy

    block_side = int(parameters['roi_xy_pix'])
    num_of_slices_P = block_side / resolution_factor
    shape_P = np.array((block_side, block_side, num_of_slices_P)).astype(np.int32)

    if _verb:
        print('\n\n*** Estimate_local_disarray()')
        print('R settings:')
        print('> R.shape: ', R.shape)
        print('> R[ev].shape: ', R['ev'].shape)
        print('> Rapporto fra Pixel Size (z / xy) =', resolution_factor)
        print('> Numero di slice selezionate per ogni ROI in R ({} x {}): {}'.format(block_side, block_side, num_of_slices_P))
        print('> Dimension of Parallelepiped in R:', shape_P, 'pixel  =  [{0:2.2f} {0:2.2f} {1:2.2f}] um'.format(
            block_side * res_xy, num_of_slices_P * res_z))

    # extract analysis subblock dimension from parameters
    Ng_z = parameters['local_disarray_z_side'] if parameters['local_disarray_z_side'] > 2 else 2
    Ng_xy = parameters['local_disarray_xy_side']
    neighbours_lim = parameters['neighbours_lim'] if parameters['neighbours_lim'] > 3 else 3

    # check if value is valid
    if Ng_xy == 0:
        Ng_xy = Ng_z * resolution_factor
    elif Ng_xy < 2:
        Ng_xy = 2

    if _verb:
        print('Disarray settings:')
        print('> Grane side in XY: ', Ng_xy)
        print('> Grane side in Z: ', Ng_z)
        print('> neighbours_lim for disarray: ', neighbours_lim)

    # shape of grane of analysis
    shape_G = (int(Ng_xy), int(Ng_xy), int(Ng_z))

    # iteration long each axis
    iterations = tuple(np.ceil(np.array(R.shape) / np.array(shape_G)).astype(np.uint32))
    if _verb: print('\n\n> Expected iterations: ', iterations)

    # define global matrix that contains global disarray
    matrix_of_disarray = np.zeros(iterations).astype(np.float32)

    if _verb: print('\n *** Start elaboration...')
    _i = 0

    for z in range(iterations[2]):
        for r in range(iterations[0]):
            for c in range(iterations[1]):

                print('iter: {0:3.0f} - (z, r, c): ({1}, {2} , {3})'.format(_i, z, r, c))
                if _verb_deep: print('\n\n\n\n')

                # grane extraction from R
                start_coord = create_coord_by_iter(r, c, z, shape_G)
                slice_coord = create_slice_coordinate(start_coord, shape_G)
                grane = R[slice_coord]  # (Gy, Gx, Gz)
                if _verb_deep:
                    print(' 0 - grane -> ', end='')
                    print(grane.shape)

                # N = Gy*Gx*Gz = n' of orientation blocks
                # (N, 3, 3)
                grane_reshaped = np.reshape(grane, np.prod(grane.shape))
                if _verb_deep:
                    print(' 1 - grane_reshaped -> ', end='')
                    print(grane_reshaped.shape)

                valid_cells = np.count_nonzero(grane_reshaped['cell_info'])
                if _verb_deep:
                    print('valid_cells --> ', valid_cells)
                    print(' valid rows: -> ', grane_reshaped['cell_info'])
                    print(' grane_reshaped[\'cell_info\'].shape:', grane_reshaped['cell_info'].shape)

                if valid_cells > parameters['neighbours_lim']:

                    # estraggo autovettore selezionato da tutte le celle
                    # (N) -> (N x 3) (select eigenvector with index 'ev_index')
                    coord = grane_reshaped['ev'][:, :, ev_index]
                    if _verb_deep:
                        print(' 2 - coord --> ', coord.shape)
                        print(coord)

                    # for print components and lin.norm of every vectors
                    if _verb_deep:
                        for iv in range(coord.shape[0]):
                            print(iv, ':', coord[iv, :], ' --> norm:', np.linalg.norm(coord[c, :]))

                    # extract only non-zero vectors:
                    # mask = np.all(np.abs(b) == 0, axis=1)
                    # valid_coords = coord[mask]  # valid rows
                    # sopra va bene, ma l'info ce l'ho già:
                    valid_coords = coord[grane_reshaped['cell_info']]
                    if _verb_deep:
                        print(' VALILD COORDS:', valid_coords.shape)
                        print(valid_coords)

                    # take a random versor (for example, the first)
                    v1 = valid_coords[0, :]

                    ## move all the vectors in the same direction
                    # (by checking the positive or negative result of dot product between the first versor and the others)
                    for i in range(valid_coords.shape[0]):
                        scalar = np.dot(v1, valid_coords[i])
                        if scalar < 0:
                            # change the direction of i-th versor
                            if _verb_deep: print(valid_coords[i], ' --->>', -valid_coords[i])
                            valid_coords[i] = -valid_coords[i]

                    # alignment degree: module of the average vector
                    alignment = np.linalg.norm(np.average(valid_coords, axis=0))
                    if _verb_deep:
                        print('alignment: ', alignment)

                    # define local_disarray degree
                    local_disarray = 100 * (1 - alignment)
                    if _verb_deep:
                        print('local_disarray: ', local_disarray)

                    # save it in each block of this portion (grane) for future statistics and plot
                    R[slice_coord]['local_disarray'] = local_disarray  # dev_tot

                    # and save it in the matrix of local_disarray
                    if _verb_deep:
                        print('saving.. rcz ({},{},{})'.format(r, c, z))

                    matrix_of_disarray[r, c, z] = local_disarray

                else:
                    R[slice_coord]['local_disarray'] = -1.  # assumption that isolated quiver have no disarray
                    matrix_of_disarray[r, c, z] = -1

    return matrix_of_disarray, shape_G, R


def main(parser):
    args = parser.parse_args()

    # Extract input information
    R_path = manage_path_argument(args.R_path)
    parameter_filename = args.parameters_filename[0]

    # extract filenames and folders
    R_name = os.path.basename(R_path)
    process_folder = os.path.basename(os.path.dirname(R_path))
    base_path = os.path.dirname(os.path.dirname(R_path))
    parameter_filepath = os.path.join(base_path, process_folder, parameter_filename)
    R_prefix = R_name.split('.')[0]

    # print some informations
    print('\n\n*** Local Disarray estimation on R matrix ***\n')
    print(' > R path:', R_path)
    print(' > process folder:', process_folder)
    print(' > base path:', base_path)
    print(' > Parameter filename : ', parameter_filename)
    print(' > Parameter filepath : ', parameter_filepath)
    print()

    # load R
    R = np.load(R_path)

    print('> R shape (r, c, z): ', R.shape)
    print('\n> R_dtype: \n', R.dtype)

    # extract parameters
    param_names = ['roi_xy_pix',
                   'px_size_xy', 'px_size_z',
                   'threshold_on_cell_ratio',
                   'local_disarray_xy_side',
                   'local_disarray_z_side',
                   'neighbours_lim']

    parameters = extract_parameters(parameter_filepath, param_names, _verb=True)

    matrix_of_disarrays, shape_G = estimate_local_disarry(R, parameters, ev_index=2, _verb=True, _verb_deep=False)

    # saving in numpy.file
    disarray_numpy_filename = 'MatrixDisarray_{}_G({},{},{})_limNeig{}.npy'.format(
        R_prefix,
        int(shape_G[0]), int(shape_G[1]), int(shape_G[2]),
        int(parameters['neighbours_lim']))
    print('\nMatrix of Disarray saved in:')
    print('>', os.path.join(base_path, process_folder))
    print('with name:', disarray_numpy_filename)
    np.save(os.path.join(base_path, process_folder, disarray_numpy_filename), matrix_of_disarrays)

    # extract valid disarray values
    disarray_values = matrix_of_disarrays[matrix_of_disarrays != -1]

    # create results strings
    results = list()
    results.append('\n \n *** Results of statistical analysis of Disarray on accepted points. \n')
    results.append('> Disarray (%):= 100 * (1 - alignment)\n')
    results.append('> Matrix of disarray shape: {}'.format(matrix_of_disarrays.shape))
    results.append('> Valid disarray values: {}'.format(disarray_values.shape))
    results.append('\n> Disarray mean: {0:0.2f}%'.format(np.mean(disarray_values)))
    results.append('> Disarray std: {0:0.2f}% '.format(np.std(disarray_values)))
    results.append('> Disarray (min, MAX)%: ({0:0.2f}, {1:0.2f})'.format(np.min(disarray_values), np.max(disarray_values)))

    # create results.txt filepath
    results_filename = 'results_disarray_by_{}_G({},{},{})_limNeig{}.txt'.format(
        R_prefix,
        int(shape_G[0]), int(shape_G[1]), int(shape_G[2]),
        int(parameters['neighbours_lim']))
    results_txt_filepath = os.path.join(base_path, process_folder, results_filename)

    with open(results_txt_filepath, 'w') as txt:
        for r in results:
            print(r)
            txt.write(r + '\n')

# ================================ END MAIN () ================================================


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Disarray analysis on Orientation vectors matrix R.npy')
    my_parser.add_argument('-r', '--R-path', nargs='+',
                           help='absolut path of 'R' numpy files containing the orientation vectors ', required=False)
    my_parser.add_argument('-p', '--parameters-filename', nargs='+',
                           help='filename of parameters.txt file (in the same folder of stack)', required=False)
    main(my_parser)

