# script che elabora il disarray (vettore locale medio -> modulo -> (1 - modulo)
# da una matrice R già esistente.
# NOTA BENE - è fatto per le matrici R delle strip - allora il campo dentro R si chiam local_disordersi chiamano DISORDER

import numpy as np
import argparse
import os

# from custom_tool_kit import manage_path_argument, create_coord_by_iter, create_slice_coordinate, \
#     search_value_in_txt, pad_dimension, write_on_txt, Bcolors
#

#
# from disarray_tools import estimate_local_disarray, save_in_numpy_file, compile_results_strings, \
#     Param, Mode, Cell_Ratio_mode, statistics_base, create_R, structure_tensor_analysis_3d, \
#     sigma_for_uniform_resolution, downsample_2_zeta_resolution, CONST

from custom_tool_kit import create_coord_by_iter, create_slice_coordinate, all_words_in_txt, \
    search_value_in_txt, Bcolors, extract_parameters, shape_P_from_parameters, create_fldr
from custom_image_base_tool import  plot_histogram, plot_map_and_save
from disarray_tools import estimate_local_disarray, Mode, save_in_numpy_file, CONST



def main(parser):

    # read args from console
    args                = parser.parse_args()
    R_filepath          = args.source_R[0]
    parameters_filename = args.source_P[0]
    _save_hist          = args.histogram
    _save_log_hist      = args.log_histogram
    _save_maps          = args.maps

    # extract paths and filenames
    base_path          = os.path.dirname(os.path.dirname(R_filepath))   # .../data/CTRL
    process_folder     = os.path.basename(os.path.dirname(R_filepath))  # .../data/CTRL/Sample1
    parameters_filepath = os.path.join(base_path, process_folder, parameters_filename)   # .../data/CTRL/Sample1/parameters.txt
    R_filename         = os.path.basename(R_filepath)
    R_prefix           = R_filename.split('.')[0]

    # output paths
    if _save_log_hist or _save_hist:
        hist_plots_path = os.path.join(base_path, process_folder, 'histograms')
        create_fldr(hist_plots_path)
    if _save_maps:
        maps_path = os.path.join(base_path, process_folder, 'maps')
        create_fldr(maps_path)

    print(Bcolors.OKBLUE + '\n\n*** Local Disarray Evaluation by Orientation Matrix R ***\n' + Bcolors.ENDC)

    # load R numpy file
    R = np.load(R_filepath)
    print(' > R matrix loaded from: \n', R_filepath)
    print(' > with shape (r, c, z): ', R.shape)

    # extract parameters
    param_names = ['roi_xy_pix',
                   'px_size_xy', 'px_size_z',
                   'mode_ratio', 'threshold_on_cell_ratio',
                   'local_disarray_xy_side',
                   'local_disarray_z_side',
                   'neighbours_lim',
                   'fwhm_xy', 'fwhm_z']

    parameters = extract_parameters(filepath=parameters_filepath, param_names=param_names)
    shape_P = shape_P_from_parameters(parameters)

    print(' > Parameters loaded from: ', parameters_filename)
    for (p, v) in parameters.items(): print(' > {} : {}'.format(p, v))

    print('\n > Start to evaluating local disarray and local Fractional Anisotropy...')
    # the function estimate local disarrays and fractional anisotropy and write these values also inside R
    mtrx_of_disarrays, mtrx_of_local_fa, shape_G, R = estimate_local_disarray(R, parameters, ev_index=2, _verb=False)

    print('Local Disarray estimated inside result Matrix')
    print(' - grane (r, c, z) used: ({}, {}, {})'.format(shape_G[0], shape_G[1], shape_G[2]))

    # save disarray matrices in a numpy file
    disarray_np_filename = dict()
    for mode in [att for att in vars(Mode) if str(att)[0] is not '_']:
        disarray_np_filename[getattr(Mode, mode)] = save_in_numpy_file(
                                                            mtrx_of_disarrays[getattr(Mode, mode)],
                                                            R_prefix, shape_G,
                                                            parameters, base_path, process_folder,
                                                            data_prefix='MatrixDisarray_{}_'.format(mode))

    # save numpy file of fractional anisotropy
    fa_np_filename = save_in_numpy_file(mtrx_of_local_fa, R_prefix, shape_G, parameters,
                                        base_path, process_folder, data_prefix='FA_local_')

    print('\n> Matrix of Disarray and Fractional Anisotropy saved in:')
    print('> {}'.format(os.path.join(base_path, process_folder)))
    print('with name: \n > {}\n > {}\n > {}\n'.format(disarray_np_filename[Mode.ARITH],
                                                      disarray_np_filename[Mode.WEIGHT],
                                                      fa_np_filename))

    if _save_hist:
        print('> Histogram plots are saved in:')

        # zip values, description and filename
        for (mtrx, lbl, np_fname) in zip([mtrx_of_disarrays[Mode.ARITH], mtrx_of_disarrays[Mode.WEIGHT], mtrx_of_local_fa],
                                         ['Local Disarray % (Arithmetic mean)', 'Local Disarray % (Weighted mean)',
                                          'Local Fractional Anisotropy %'],
                                         [disarray_np_filename[Mode.ARITH], disarray_np_filename[Mode.WEIGHT],
                                          fa_np_filename]):

            # extract only valid values (different of INV = -1)
            valid_values = mtrx[mtrx != CONST.INV]
            # convert FA to percentage
            if lbl == 'Local Fractional Anisotropy %':
                valid_values = 100 * valid_values

            # create path name
            hist_fname = '.'.join(np_fname.split('.')[:-1]) + '.tiff'  # remove '.npy' and add .tiff
            hist_filepath = os.path.join(hist_plots_path, hist_fname)
            # create histograms and save they as images
            plot_histogram(valid_values, xlabel=lbl, ylabel='Sub-Volume occurrence', filepath=hist_filepath)
            print(' > {}'.format(hist_filepath))

    if _save_log_hist:
        print('> Histogram plots of LOG values are saved in:')
        # zip values, description and filename
        for (mtrx, lbl, np_fname) in zip([mtrx_of_disarrays[Mode.ARITH], mtrx_of_disarrays[Mode.WEIGHT]],
                                         ['LOG of Local Disarray % (Arithmetic mean)', 'LOG of Local Disarray % (Weighted mean)'],
                                         [disarray_np_filename[Mode.ARITH], disarray_np_filename[Mode.WEIGHT]]):

            # extract only valid values (different of INV = -1)
            valid_values = mtrx[mtrx != CONST.INV]
            log_values = np.log(valid_values)

            # create path name
            hist_fname = '.'.join(np_fname.split('.')[:-1]) + '_LOG_.tiff'  # remove '.npy' and add LOG.tiff
            hist_filepath = os.path.join(hist_plots_path, hist_fname)
            # create histograms and save they as images
            plot_histogram(log_values, xmin=0, xmax=5, xlabel=lbl, ylabel='Sub-Volume occurrence', filepath=hist_filepath)
            print(' > {}'.format(hist_filepath))

    if _save_maps:

        print('\n> Disarray and FA plots are saved in:')
        # convert fractional anisotropy in percentual
        mtrx_of_local_fa_perc = 100 * mtrx_of_local_fa
        # create and save frame for each data (disarrays and FA)

        for (mtrx, np_fname) in zip(
                [mtrx_of_disarrays[Mode.ARITH], mtrx_of_disarrays[Mode.WEIGHT], mtrx_of_local_fa_perc],
                [disarray_np_filename[Mode.ARITH], disarray_np_filename[Mode.WEIGHT], fa_np_filename]):
            # plot frames and save they in a sub_folder (map_path)
            map_path = plot_map_and_save(mtrx, np_fname, maps_path,
                                         parameters['px_size_xy'], parameters['px_size_z'], shape_G, shape_P)
            print('> {}'.format(map_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimation of Local Disarray from Orientation vectors')
    parser.add_argument('-r', '--source-R', nargs='+', help='Filepath of the Orientation matrix to load', required=False)
    parser.add_argument('-p', '--source-P', nargs='+', help='Filename of Parameters .txt file to load', required=False)
    parser.add_argument('-i', action='store_true', default=True, dest='histogram',
                                                  help='save histograms of disattay values as image')
    parser.add_argument('-l', action='store_true', default=True, dest='log_histogram',
                        help='save histograms of log of disarray values as image')
    parser.add_argument('-m', action='store_true', default=True, dest='maps',
                                                  help='save maps of disarray and FA as frames')
    main(parser)
