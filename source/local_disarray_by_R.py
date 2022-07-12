
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

from custom_tool_kit import Bcolors, extract_parameters, shape_P_from_parameters, create_fldr, \
    collect_list_of_values_in_the_txt
from custom_image_base_tool import  plot_histogram, plot_map_and_save
from disarray_tools import estimate_local_disarray, Mode, save_in_numpy_file, CONST

#not general, useful to clean main code
def plot_and_save_histogram(mtrx, lbl, np_fname, hist_plots_path, _log=False):

    # create filename name
    hist_fname = '.'.join(np_fname.split('.')[:-1])  # remove '.npy'

    # extract only valid values (different of INV = -1)
    valid_values = mtrx[mtrx != CONST.INV]

    # convert FA to percentage
    if lbl == 'Local Fractional Anisotropy %':
        valid_values = 100 * valid_values

    if _log:
        valid_values = np.log(valid_values)
        hist_fname = hist_fname + '_LOG.tiff'
        xmin, xmax = 0, 5
    else:
        hist_fname = hist_fname + '.tiff'
        xmin, xmax = 0, 100

    hist_filepath = os.path.join(hist_plots_path, hist_fname)
    # create histograms and save they as images
    plot_histogram(valid_values, xmin=xmin, xmax=xmax,
                   xlabel=lbl, ylabel='Sub-Volume occurrence', filepath=hist_filepath)
    return hist_filepath


# not general, useful here to clean code
def compile_param_x_grane(parameters, grane):
    current_parameters = parameters.copy()

    # write the current grane
    current_parameters['local_disarray_z_side'] = grane
    current_parameters['local_disarray_xy_side'] = grane

    # write the current lim on valid neighbours vectors
    numbers_of_cells = grane**3
    min_num_valid_cells = np.floor(numbers_of_cells * parameters['neighbours_lim_perc'] / 100)
    current_parameters['neighbours_lim'] = min_num_valid_cells
    return current_parameters


def main(parser):

    # read args from console
    args                = parser.parse_args()
    R_filepath          = args.source_R[0]
    parameters_filename = args.source_P[0]
    _save_hist          = args.histogram
    _save_log_hist      = args.log_histogram
    _save_maps          = args.maps

    # collect paths and filenames
    base_path          = os.path.dirname(os.path.dirname(R_filepath))   # .../data/CTRL
    process_folder     = os.path.basename(os.path.dirname(R_filepath))  # .../data/CTRL/Sample1
    parameters_filepath = os.path.join(base_path, process_folder, parameters_filename)   # .../data/CTRL/Sample1/parameters.txt
    R_filename         = os.path.basename(R_filepath)
    R_prefix           = R_filename.split('.')[0]

    # out .txt information
    mess_strings = list()

    # print to terminal
    print(Bcolors.OKBLUE + '\n\n*** Local Disarray Evaluation by Orientation Matrix R ***\n' + Bcolors.ENDC)

    # load R numpy file
    R = np.load(R_filepath)
    mess_strings.append(' > R matrix loaded from: \n{}'.format(R_filepath))
    mess_strings.append(' > with shape (r, c, z): ({}, {}, {})'.format(R.shape[0], R.shape[1], R.shape[2]))

    # extract parameters
    param_names_to_collect = ['roi_xy_pix',
                              'px_size_xy', 'px_size_z',
                              'fwhm_xy', 'fwhm_z',
                              'neighbours_lim_perc']
    parameters = extract_parameters(filepath=parameters_filepath, param_names=param_names_to_collect)
    shape_P = shape_P_from_parameters(parameters)

    # extract list of granes from parameters.txt file
    granes_keyword = 'local_disarray_granes'
    granes = collect_list_of_values_in_the_txt(parameters_filepath, granes_keyword)
    parameters['granes'] = granes

    mess_strings.append(' > Parameters loaded from: {}'.format(parameters_filename))
    for (p, v) in parameters.items():
        mess_strings.append(' > {} : {}'.format(p, v))

    mess_strings.append('\n > Start to evaluating Local disarray and local Fractional Anisotropy...')
    for mess in mess_strings:
        print(mess)
    mess_strings.clear()

    for grane in parameters['granes']:
        # ================================ START CURRENT DISARRAY EVALUATION ==========================
        print(Bcolors.OKBLUE + 'Evaluating Disarray with Grane: {0} x {0} x {0}'.format(grane) + Bcolors.ENDC)

        # output paths
        if _save_log_hist or _save_hist:
            hist_plots_path = os.path.join(base_path, process_folder, 'histograms_G({0}x{0}x{0})'.format(grane))
            create_fldr(hist_plots_path)
        if _save_maps:
            maps_path = os.path.join(base_path, process_folder, 'maps_G({0}x{0}x{0})'.format(grane))
            create_fldr(maps_path)

        # parameters of current spatial resolution
        current_param = compile_param_x_grane(parameters, grane)

        # the function estimate local disarrays and fractional anisotropy
        mtrx_of_disarrays, mtrx_of_local_fa, shape_G, R = estimate_local_disarray(R, current_param, ev_index=2, _verb=False)
        mess_strings.append('Local Disarray estimated by R with grane (r, c, z): ({}, {}, {})'.format(shape_G[0], shape_G[1], shape_G[2]))

        # save disarray and FA matrices in numpy files
        disarray_np_filenames = dict()

        for mode in [att for att in vars(Mode) if str(att)[0] != '_']:
            disarray_np_filenames[getattr(Mode, mode)] = save_in_numpy_file(
                                                                mtrx_of_disarrays[getattr(Mode, mode)],
                                                                R_prefix, shape_G,
                                                                current_param, base_path, process_folder,
                                                                data_prefix='MatrixDisarray_{}_'.format(mode))
        # save numpy file of fractional anisotropy
        fa_np_filename = save_in_numpy_file(mtrx_of_local_fa, R_prefix, shape_G, current_param,
                                            base_path, process_folder, data_prefix='FA_local_')

        print('\n> Matrix of Disarray and Fractional Anisotropy saved in:')
        print('> {}'.format(os.path.join(base_path, process_folder)))
        print('with name: \n > {}\n > {}\n > {}\n'.format(disarray_np_filenames[Mode.ARITH],
                                                          disarray_np_filenames[Mode.WEIGHT],
                                                          fa_np_filename))
        if _save_hist:
            print('> Histogram plots are saved in:')
            # zip values, description and filename
            for (mtrx, xlbl, np_fname) in zip([mtrx_of_disarrays[Mode.ARITH], mtrx_of_disarrays[Mode.WEIGHT], mtrx_of_local_fa],
                                             ['Local Disarray % (Arithmetic mean)', 'Local Disarray % (Weighted mean)',
                                              'Local Fractional Anisotropy %'],
                                             [disarray_np_filenames[Mode.ARITH], disarray_np_filenames[Mode.WEIGHT],
                                              fa_np_filename]):
                # produce and save current histograms
                hist_filepath = plot_and_save_histogram(mtrx, xlbl, np_fname, hist_plots_path, _log=False)
                mess_strings.append(' > {}'.format(hist_filepath))

        if _save_log_hist:
            print('> Histogram plots of LOG values are saved in:')
            # zip values, description and filename
            for (mtrx, xlbl, np_fname) in zip([mtrx_of_disarrays[Mode.ARITH], mtrx_of_disarrays[Mode.WEIGHT]],
                                             ['LOG of Local Disarray % (Arithmetic mean)', 'LOG of Local Disarray % (Weighted mean)'],
                                             [disarray_np_filenames[Mode.ARITH], disarray_np_filenames[Mode.WEIGHT]]):

                log_hist_filepath = plot_and_save_histogram(mtrx, xlbl, np_fname, hist_plots_path, _log=True)
                mess_strings.append(' > {}'.format(log_hist_filepath))

        if _save_maps:

            print('\n> Disarray and FA plots are saved in:')
            # convert fractional anisotropy in percentual
            mtrx_of_local_fa_perc = 100 * mtrx_of_local_fa
            # create and save frame for each data (disarrays and FA)

            # generate maps of invalid value (blocks where disarray is not analyzed)
            mtrx_of_invalid_masks = mtrx_of_disarrays[Mode.WEIGHT] < 0  # boolean
            mtrx_of_invalid_masks = mtrx_of_invalid_masks * 255  # ldg(invalid blocks have ldg 255)
            invalid_masks_filename = 'MASK_not_analyzed_{}_G({},{},{}).npy'.format(
                R_prefix,
                int(shape_G[0]), int(shape_G[1]), int(shape_G[2]))

            for (mtrx, np_fname) in zip(
                    [mtrx_of_disarrays[Mode.ARITH], mtrx_of_disarrays[Mode.WEIGHT], mtrx_of_local_fa_perc, mtrx_of_invalid_masks],
                    [disarray_np_filenames[Mode.ARITH], disarray_np_filenames[Mode.WEIGHT], fa_np_filename, invalid_masks_filename]):
                # plot frames and save they in a sub_folder (map_path)
                map_path = plot_map_and_save(mtrx, np_fname, maps_path,
                                             parameters['px_size_xy'], parameters['px_size_z'],
                                             shape_G, shape_P, _save_MIP=True, _save_AVG=True)
                print('> {}'.format(map_path))
        # ================================== end disarray evaluation with current grane ================================

    print(Bcolors.OKBLUE + '\nFinish to evaluate Disarray on current sample. \n' + Bcolors.ENDC)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimation of Local Disarray from Orientation vectors')
    parser.add_argument('-r', '--source-R', nargs='+', help='Filepath of the Orientation matrix to load', required=False)
    parser.add_argument('-p', '--source-P', nargs='+', help='Filename of Parameters .txt file to load', required=False)
    parser.add_argument('-i', action='store_true', default=False, dest='histogram',
                                                  help='save histograms of disarray values as images (default: False)')
    parser.add_argument('-l', action='store_true', default=False, dest='log_histogram',
                        help='save histograms of log of disarray values as images (default: False)')
    parser.add_argument('-m', action='store_true', default=False, dest='maps',
                        help='save maps of disarray and FA as frames (default: False)')
    main(parser)
