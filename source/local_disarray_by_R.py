# system
import os
import argparse

# general
import numpy as np

# custom codes
from custom_tool_kit import manage_path_argument, create_coord_by_iter, create_slice_coordinate, all_words_in_txt, search_value_in_txt


class Param:
    ID_BLOCK = 'id_block'  # unique identifier of block
    CELL_INFO = 'cell_info'  # 1 if block is analyzed, 0 if it is rejected by cell_threshold
    ORIENT_INFO = 'orient_info'  # 1 if block is analyzed, 0 if it is rejected by cell_threshold
    CELL_RATIO = 'cell_ratio'  # ratio between cell voxel and all voxel of block
    INIT_COORD = 'init_coord'   # absolute coord of voxel block[0,0,0] in Volume
    EW = 'ew'   # descending ordered eigenvalues.
    EV = 'ev'   # column ev[:,i] is the eigenvector of the eigenvalue w[i].
    STRENGHT = 'strenght'   # parametro forza del gradiente (w1 .=. w2 .=. w3)
    CILINDRICAL_DIM = 'cilindrical_dim'  # dimensionalità forma cilindrica (w1 .=. w2 >> w3)
    PLANAR_DIM = 'planar_dim'  # dimensionalità forma planare (w1 >> w2 .=. w3)
    FA = 'fa'  # fractional anisotropy (0-> isotropic, 1-> max anisotropy
    LOCAL_DISARRAY = 'local_disarray'   # local_disarray
    LOCAL_DISARRAY_W = 'local_disarray_w'  # local_disarray using FA as weight for the versors


class Stat:
    # statistics
    N_VALID_VALUES = 'n_valid_values'
    MIN = 'min'
    MAX = 'max'
    AVG = 'avg'  # statistics mean
    STD = 'std'
    MEDIAN = 'median'  # statistics median
    MODE = 'mode'  # statistics mode
    MODALITY = 'modality'  # for arithmetic or weighted


class Mode:
    # to select which type of local disarray is:
    # estimated with arithmetic average
    # or
    # estimated with weighted average
    ARITH = 'arithmetic'
    WEIGHT = 'weighted'


class Bcolors:
    # to color different strings in the output0
    V = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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


def estimate_local_disarray(R, parameters, ev_index=2, _verb=True, _verb_deep=False):
    # give R and:
    # - according with the space resolution defined in 'parameters'
    # - using only the eigenvectors with index 'ev_index'
    # estimate the local_disarray (disarray in cluster of versors) in the entire R matrix.
    # Local disarray is defined as: 100 * (1 - align),
    # where local align is estimated by the module of the average vector of the local versors.
    # Two different average are used:
    # an arithmetic average (version 0)
    # a weighted average, where every vector is weighted with its Fractional Anisotropy (version 1).
    # Both versions are saved in the structured var 'matrices_of_disarrays', as
    # matrices_of_disarrays['arithmetic'] and matrices_of_disarrays['weighted']

    # extract parameters
    res_xy = parameters['px_size_xy']
    res_z = parameters['px_size_z']
    resolution_factor = res_z / res_xy

    # estimate space resolution of R
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

    # extract disarray space resolution from parameters
    Ng_z = parameters['local_disarray_z_side']
    Ng_xy = parameters['local_disarray_xy_side']
    neighbours_lim = parameters['neighbours_lim']

    # check if disarray space resolution on image plane is valid
    if Ng_xy == 0:
        Ng_xy = int(Ng_z * resolution_factor)
    elif Ng_xy < 2:
        Ng_xy = 2

    if _verb:
        print('Disarray settings:')
        print('> Grane side in XY: ', Ng_xy)
        print('> Grane side in Z: ', Ng_z)
        print('> neighbours_lim for disarray: ', neighbours_lim)

    # shape of grane of disarray analysis
    shape_G = (int(Ng_xy), int(Ng_xy), int(Ng_z))

    # iteration long each axis
    iterations = tuple(np.ceil(np.array(R.shape) / np.array(shape_G)).astype(np.uint32))
    if _verb:
        print('\n\n> Expected iterations on each axis: ', iterations)
        print('\n\n> Expected total iterations       : ', np.prod(iterations))

    # define global matrix that contains local disarrays and local FA
    # - local disarray is the disarray of the selected cluster of orientation versors
    # - local FA is the mean of the FAs of the selected cluster of orientation versors
    matrices_of_disarray = dict()
    matrices_of_disarray[Mode.ARITH] = np.zeros(iterations).astype(np.float32)
    matrices_of_disarray[Mode.WEIGHT] = np.zeros(iterations).astype(np.float32)
    matrix_of_local_fa = np.zeros(iterations).astype(np.float32)

    if _verb: print('\n *** Start elaboration...')
    if _verb_deep: print(Bcolors.V)  # open colored session
    _i = 0
    for z in range(iterations[2]):
        for r in range(iterations[0]):
            for c in range(iterations[1]):
                if _verb_deep:
                    print(' *** DEBUGGING MODE ACTIVATED ***')
                    print('\n\n\n\n')

                print(Bcolors.WARNING +
                      'iter: {0:3.0f} - (z, r, c): ({1}, {2} , {3})'.format(_i, z, r, c) +
                      Bcolors.V)

                # grane extraction from R
                start_coord = create_coord_by_iter(r, c, z, shape_G)
                slice_coord = tuple(create_slice_coordinate(start_coord, shape_G))
                grane = R[slice_coord]  # extract a sub-volume called 'grane' with size (Gy, Gx, Gz)
                if _verb_deep:
                    print(' 0 - grane -> ', end='')
                    print(grane.shape)

                # N = Gy*Gx*Gz = n' of orientation blocks
                # (gx, gy, gz) --> (N,)
                grane_reshaped = np.reshape(grane, np.prod(grane.shape))
                if _verb_deep:
                    print(' 1 - grane_reshaped -> ', end='')
                    print(grane_reshaped.shape)

                n_valid_cells = np.count_nonzero(grane_reshaped[Param.CELL_INFO])
                if _verb_deep:
                    print(' valid_cells --> ', n_valid_cells)
                    print(' valid rows: -> ', grane_reshaped[Param.CELL_INFO])
                    print(' grane_reshaped[\'cell_info\'].shape:', grane_reshaped[Param.CELL_INFO].shape)

                if n_valid_cells > parameters['neighbours_lim']:

                    # (N) -> (N x 3) (select eigenvector with index 'ev_index' from all the N cells)
                    coord = grane_reshaped[Param.EV][:, :, ev_index]
                    if _verb_deep:
                        print(' 2 - coord --> ', coord.shape)
                        print(coord)

                    # extract fractional anisotropy (N)
                    fa = grane_reshaped[Param.FA]

                    # for print components, lin.norm and FA of every versors (iv = index_of_versor)
                    if _verb_deep:
                        for iv in range(coord.shape[0]):
                            print(iv, ':', coord[iv, :],
                                  ' --> norm:', np.linalg.norm(coord[iv, :]),
                                  ' --> FA: ', fa[iv])

                    # select only versors and FAs from valid cells:
                    valid_coords = coord[grane_reshaped[Param.CELL_INFO]]
                    valid_fa = fa[grane_reshaped[Param.CELL_INFO]]
                    if _verb_deep:
                        print(' valid coords - ', valid_coords.shape, ' :')
                        print(valid_coords)
                        print(' valid FAs - ', valid_fa.shape, ' :')
                        print(valid_fa)

                    # order the valid versors by their FA (descending)
                    # [use '-fa' for obtain descending order]
                    ord_coords = valid_coords[np.argsort(-valid_fa)]
                    ord_fa = valid_fa[np.argsort(-valid_fa)]

                    # take the first versor (biggest FA) to
                    # move all other versors in the same half-space
                    v1 = ord_coords[0, :]

                    # move all other versors in the same direction
                    # (by checking the positive or negative result of dot product between
                    # the first versor and the others)
                    if _verb_deep: print('Check if versors have congruent direction (same half-space)')
                    for iv in range(1, ord_coords.shape[0]):
                        scalar = np.dot(v1, ord_coords[iv])
                        if scalar < 0:
                            # change the direction of i-th versor
                            if _verb_deep: print(ord_coords[iv], ' --->>', -ord_coords[iv])
                            ord_coords[iv] = -ord_coords[iv]

                    # for print definitive components, lin.norm and FA of every versors
                    if _verb_deep:
                        print('Definitive versors component in the same half-space:')
                        for iv in range(ord_coords.shape[0]):
                            print(iv, ':', ord_coords[iv, :],
                                  ' --> norm:', np.linalg.norm(ord_coords[iv, :]),
                                  ' --> FA: ', ord_fa[iv])

                    if _verb_deep:
                        print('np.average(ord_coords): \n', np.average(ord_coords, axis=0))
                        print('np.average(ord_coords, weight=fa): \n',
                              np.average(ord_coords, axis=0, weights=ord_fa))

                    # alignment degree: module of the average vector.
                    # The averages estimated are both arithmetical and weighted with the FA
                    alignment = dict()
                    alignment[Mode.ARITH] = np.linalg.norm(np.average(ord_coords, axis=0))
                    alignment[Mode.WEIGHT] = np.linalg.norm(np.average(ord_coords, axis=0, weights=ord_fa))
                    if _verb_deep:
                        print('alignment[Mode.ARITH] : ', alignment[Mode.ARITH])
                        print('alignment[Mode.WEIGHT]: ', alignment[Mode.WEIGHT])

                    # define local_disarray degree
                    local_disarray = dict()
                    local_disarray[Mode.ARITH] = 100 * (1 - alignment[Mode.ARITH])
                    local_disarray[Mode.WEIGHT] = 100 * (1 - alignment[Mode.WEIGHT])

                    # save the disarray in each block of this portion (grane)
                    # for future statistics and plot
                    R[slice_coord][Param.LOCAL_DISARRAY] = local_disarray[Mode.ARITH]
                    R[slice_coord][Param.LOCAL_DISARRAY_W] = local_disarray[Mode.WEIGHT]

                    # estimate average Fractional Anisotropy
                    # and save results into matrix of local disarray
                    matrix_of_local_fa[r, c, z] = np.mean(ord_fa)
                    matrices_of_disarray[Mode.ARITH][r, c, z] = local_disarray[Mode.ARITH]
                    matrices_of_disarray[Mode.WEIGHT][r, c, z] = local_disarray[Mode.WEIGHT]
                    if _verb_deep:
                        print('saving.. rcz:({},{},{}):'.format(r, c, z))
                        print('local_disarray[Mode.ARITH] : ', local_disarray[Mode.ARITH])
                        print('local_disarray[Mode.WEIGHT]: ', local_disarray[Mode.WEIGHT])
                        print('mean Fractional Anisotropy : ', matrix_of_local_fa[r, c, z])

                else:
                    # Assign invalid value (-1)
                    # (assume that isolated quiver have no disarray)
                    R[slice_coord][Param.LOCAL_DISARRAY] = -1
                    R[slice_coord][Param.LOCAL_DISARRAY_W] = -1
                    matrices_of_disarray[Mode.ARITH][r, c, z] = -1
                    matrices_of_disarray[Mode.WEIGHT][r, c, z] = -1
                    matrix_of_local_fa[r, c, z] = -1

                # end iteration
                _i += 1

    # close colored session
    if _verb_deep:
        print(Bcolors.ENDC)

    return matrices_of_disarray, matrix_of_local_fa, shape_G, R


def save_in_numpy_file(matrix_of_disarrays, R_prefix, shape_G, parameters,
                       base_path, process_folder, data_prefix=''):

    numpy_filename_endname = '{}_G({},{},{})_limNeig{}.npy'.format(
        R_prefix,
        int(shape_G[0]), int(shape_G[1]), int(shape_G[2]),
        int(parameters['neighbours_lim']))

    disarray_numpy_filename = data_prefix + numpy_filename_endname
    np.save(os.path.join(base_path, process_folder, disarray_numpy_filename), matrix_of_disarrays)
    return disarray_numpy_filename


# # TODO DA RIMUOVERE E SOSTITUIRE CON STATISTICS
# def statistic_strings_of_valid_values(matrix, weights=None, _verb=False):
#
#     # TODO HARDCODED
#     _verb = True
#
#     if _verb:
#         print(Bcolors.V)
#         if weights is not None:
#             print('* Statistic estimation with weighted average')
#         else:
#             print('* Statistic estimation with arithmetic average')
#
#     if _verb:
#         print('matrix.shape: ', matrix.shape)
#         if weights is not None: print('weights.shape: ', weights.shape)
#
#     stat = dict()
#
#     # extract valid values (it becomes 1-axis vector)
#     # TODO -> FAKE ! QUI LA fa MI ARRIVA 0.0 SE IL BLOCCO ERA SENZA INFO! NON -1!  <<-------- - - - -    PAY ATTENTION
#     valid_values = matrix[matrix != 0]
#
#     if weights is not None:
#         # todo IDEM, SELEZIONARE PER BENE SOLO LE CELLE VALIDE (O GLI PASSO SOLO QUELLE??) pensarci !!! <<-------- - - - - -  - - - - -  -  -  -  -  -  -  -  -  PAY ATTENTION
#         weights = np.ndarray.flatten(weights)  # to fit valid_values shape
#     if _verb:
#         print('valid values extracted')
#
#     if _verb:
#         print('valid_values.shape: ', valid_values.shape)
#         if weights is not None: print('weights.shape: ', weights.shape)
#
#     # collect shape, min and max
#     stat['n_valid_values'] = valid_values.shape[0]
#     stat['min'] = valid_values.min()
#     stat['max'] = valid_values.max()
#
#     # collect avg and std
#     if weights is not None:
#         stat['avg'] = np.average(valid_values,
#                                  axis=0,
#                                  weights=weights)
#         stat['std'] = np.sqrt(np.average((valid_values - stat['avg']) ** 2,
#                                          axis=0,
#                                          weights=weights))
#     else:
#         stat['avg'] = np.average(valid_values)
#         stat['std'] = np.std(valid_values)
#
#     if _verb:
#         print('avg:, ', stat['avg'])
#         print('std:, ', stat['std'])
#         print(Bcolors.ENDC)
#     return stat


def compile_results_strings(matrix, name, stats, mode='none_passed', ext=''):
    """

    :param matrix: input values
    :param name: name of the parameter of matrix
    :param stats: statistic results on matrix
    :param mode: aritm or weighted
    :param ext: extension of the results (%, um, none..)
    :return: strings: list of strings
    """
    strings = list()

    strings.append('\n \n *** Results of statistical analysis of {} on accepted points. \n'.format(name))
    strings.append('> Matrix of {} with shape: {}'.format(name, matrix.shape))
    strings.append('> Number of valid values: {}'.format(stats[Stat.N_VALID_VALUES]))
    strings.append('> Modality of statistical evaluation: {}\n'.format(mode))
    strings.append('> Statistical Results:')

    # generation of a string for each statistical parameters
    for att in [att for att in vars(Stat) if str(att)[0] is not '_']:
        # print(att)
        # print(getattr(Stat, att))
        # print(stats)
        # print('{0}'.format(getattr(Stat, att)))
        # print('{0:0.2f}'.format(stats[getattr(Stat, att)]))
        # print('{0}'.format(ext))

        # check if the parameter contains a string (ex: 'ARIITH' or 'WEIGHT')
        if isinstance(stats[getattr(Stat, att)], str):
            strings.append(' - {0}: {1}'.format(getattr(Stat, att), stats[getattr(Stat, att)]))

        elif att == Stat.N_VALID_VALUES:
            # print integer and without extension
            strings.append(' - {0}: {1}'.format(getattr(Stat, att), stats[getattr(Stat, att)]))
        else:
            strings.append(' - {0}: {1:0.2f}{2}'.format(getattr(Stat, att), stats[getattr(Stat, att)], ext))
    return strings


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

    #extract parameters
    parameters = extract_parameters(parameter_filepath, param_names, _verb=True)

    # estimate disarray
    matrices_of_disarrays, matrix_of_local_fa, shape_G, R = estimate_local_disarray(R, parameters, ev_index=2, _verb=True, _verb_deep=False)

    # save numpy file of both disarrays matrix (caculated with arithmetic and weighted average)
    # estraggo lista degli attributi della classe Mode
    # e scarto quelli che cominciao con '_' perchè saranno moduli
    disarray_numpy_filename = dict()
    for mode in [att for att in vars(Mode) if str(att)[0] is not '_' ]:
        disarray_numpy_filename[mode] = save_in_numpy_file(
                                                            matrices_of_disarrays[mode], R_prefix, shape_G,
                                                            parameters, base_path, process_folder,
                                                            data_prefix='MatrixDisarray_{}_'.format(mode))
    # save numpy file of fractional anisotropy
    fa_numpy_filename = save_in_numpy_file(
                                            matrix_of_local_fa, R_prefix, shape_G, parameters,
                                            base_path, process_folder, data_prefix='FA_local_')

    print('\nMatrix of Disarray and Fractional Anisotropy saved in:')
    print('>', os.path.join(base_path, process_folder))
    print('with name: '
          '\n >', disarray_numpy_filename[Mode.ARITH],
          '\n >', disarray_numpy_filename[Mode.WEIGHT],
          '\n >', fa_numpy_filename)

    # evaluated statistics for both disarray (arithm and weighted)
    disarray_ARITM_stats = statistic_strings_of_valid_values(matrices_of_disarrays[Mode.ARITH])
    disarray_WEIGHT_stats = statistic_strings_of_valid_values(matrices_of_disarrays[Mode.WEIGHT],
                                                              weights=matrix_of_local_fa)
    fa_stats = statistic_strings_of_valid_values(matrix_of_local_fa)

    # create disarray results strings
    s1 = compile_results_strings(matrices_of_disarrays[Mode.ARITH], disarray_ARITM_stats,
                                 fa_stats, mode=Mode.ARITH)
    s2 = compile_results_strings(matrices_of_disarrays[Mode.WEIGHT], disarray_WEIGHT_stats,
                                 fa_stats, mode=Mode.ARITH)
    disarray_and_fa_results_strings = s1 + s2

    # create disarray_results_strings.txt filepath
    disarray_txt_results_filename = 'results_disarray_by_{}_G({},{},{})_limNeig{}.txt'.format(
        R_prefix,
        int(shape_G[0]), int(shape_G[1]), int(shape_G[2]),
        int(parameters['neighbours_lim']))
    disarray_txt_results_filepath = os.path.join(base_path, process_folder, disarray_txt_results_filename)

    with open(disarray_txt_results_filepath, 'w') as txt:
        for r in disarray_and_fa_results_strings:
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

