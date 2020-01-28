''' #####################################################################
# SIMPLE EXAMPLE TO ORDER THE EIGENVECTOR AND USE IT

ew = np.array([10, 1, 100])
ev = np.random.randint(low=-10, high=10, size=9, dtype='l').reshape(3, 3)
ev[:, 0] = ev[:, 0] * 10
ev[:, 1] = ev[:, 1] * 1
ev[:, 2] = ev[:, 2] * 100
print(ew)
print(ev)

# ordino autovalori e autovettori t.c. w0 > w1 > w2
order = np.argsort(ew)[::-1];  # decrescent order
ew = np.copy(ew[order]);
ev = np.copy(ev[:, order]);
print()
print(ew)
print(ev)

print()
print(ew[0], ' --> ', ev[:, 0])
print(ew[1], ' --> ', ev[:, 1])
print(ew[2], ' --> ', ev[:, 2])

print('... rotation?')
ev_rotated = np.zeros_like(ev)
for axis in range(ev.shape[1]):
    ev_rotated[:, axis] = check_in_upper_semisphere(ev[:, axis])

print(ew[0], ' --> ', ev_rotated[:, 0])
print(ew[1], ' --> ', ev_rotated[:, 1])
print(ew[2], ' --> ', ev_rotated[:, 2])
##################################################################### '''

# system
import os
import time
import argparse

# general
import numpy as np

from scipy import stats as scipy_stats
from zetastitcher import InputFile

# custom codes
from custom_tool_kit import manage_path_argument, create_coord_by_iter, create_slice_coordinate, \
    search_value_in_txt, pad_dimension, write_on_txt, Bcolors
from custom_image_base_tool import normalize, print_info, plot_histogram, plot_map_and_save
from disarray_tools import estimate_local_disarray, save_in_numpy_file, compile_results_strings, \
    Param, Mode, Cell_Ratio_mode, statistics_base, create_R, structure_tensor_analysis_3d, \
    sigma_for_uniform_resolution, downsample_2_zeta_resolution, CONST




# =================================================== MAIN () ================================================
def main(parser):

    # INPUT HARDCODED FOR DEBUG ==================================================================================
    # source_path = '/home/francesco/LENS/ST_analysis_tests/test_vasi/200116_test_vasi_FA_3.1/stack.tif'
    # parameter_filename = 'parameters_vessels.txt'
    # _verbose = False
    # _deep_verbose = False
    # _save_csv = True
    # _save_hist = True
    # _save_maps = True
    # if _verbose:
    #     print(Bcolors.FAIL + ' *** DEBUGGING MODE *** ' + Bcolors.ENDC)
    # ============================================================================================================


    ## Extract input information FROM TERMINAL =========
    args = parser.parse_args()
    R_filepath = manage_path_argument(args.R_path)
    parameter_filename = args.parameters_filename[0]
    _verbose = args.verbose
    _deep_verbose = args.deep_verbose
    _save_csv = args.csv
    _save_hist = args.histogram
    _save_maps = args.maps
    if _verbose:
        print(Bcolors.FAIL + ' *** VERBOSE MODE *** ' + Bcolors.ENDC)
    if _deep_verbose:
        print(Bcolors.FAIL + ' *** DEBUGGING MODE *** ' + Bcolors.ENDC)
    ## =================================================

    # extract filenames and folders
    R_filename = os.path.basename(R_filepath)
    process_folder = os.path.basename(os.path.dirname(R_filepath))
    base_path = os.path.dirname(os.path.dirname(R_filepath))
    parameter_filepath = os.path.join(base_path, process_folder, parameter_filename)
    stack_prefix = R_filepath.split('.')[1]

    # create introductiveme informations
    mess_strings = list()
    mess_strings.append(Bcolors.OKBLUE + '\n\n*** Disarray Analysis ***\n' + Bcolors.ENDC)
    mess_strings.append(' > R matrix : {}'.format(R_filename))
    mess_strings.append(' > base path: {}'.format(base_path))
    mess_strings.append(' > Parameter filename: {}'.format(parameter_filename))
    mess_strings.append(' > Parameter filepath: {}'.format(parameter_filepath))
    mess_strings.append('')
    mess_strings.append(' > PREFERENCES:')
    mess_strings.append('  - _verbose {}'.format(_verbose))
    mess_strings.append('  - _deep_verbose {}'.format(_deep_verbose))
    mess_strings.append('  - _save_csv {}'.format(_save_csv))
    mess_strings.append('  - _save_hist {}'.format(_save_hist))
    mess_strings.append('  - _save_maps {}'.format(_save_maps))

    # extract parameters
    param_names = ['roi_xy_pix',
                   'px_size_xy', 'px_size_z',
                   'mode_ratio', 'threshold_on_cell_ratio',
                   'local_disarray_xy_side',
                   'local_disarray_z_side',
                   'neighbours_lim',
                   'fwhm_xy','fwhm_z']

    param_values = search_value_in_txt(parameter_filepath, param_names)

    # create dictionary of parameters
    parameters = {}
    mess_strings.append('\n\n*** Parameters used:')
    mess_strings.append(' > Parameters extracted from {}\n'.format(parameter_filename))
    for i, p_name in enumerate(param_names):
        parameters[p_name] = float(param_values[i])
        mess_strings.append('> {} - {}'.format(p_name, parameters[p_name]))

    # Parameters of Acquisition System:
    # ratio between pixel size in z and xy
    ps_ratio = parameters['px_size_z'] / parameters['px_size_xy']

    # analysis block dimension in z-axis
    shape_P = np.array((int(parameters['roi_xy_pix']),
                        int(parameters['roi_xy_pix']),
                        int(parameters['roi_xy_pix'] / ps_ratio))).astype(np.int32)

    mess_strings.append('\n *** Analysis configuration')
    mess_strings.append(' > Rapporto fra Pixel Size (z / xy) = {0:0.2f}'.format(ps_ratio))
    mess_strings.append(' > Numero di slice selezionate per ogni ROI ({} x {}): {}'.format(
        shape_P[0], shape_P[1], shape_P[2]))
    mess_strings.append(' > Dimension of Parallelepiped: ({0},{1},{2}) pixel  ='
                        '  [{3:2.2f} {4:2.2f} {5:2.2f}] um'.format(
        shape_P[0], shape_P[1], shape_P[2],
        shape_P[0] * parameters['px_size_xy'],
        shape_P[1] * parameters['px_size_xy'],
        shape_P[2] * parameters['px_size_z']))

    # print to screen
    for s in mess_strings:
        print(s)
    # clear list of strings
    mess_strings.clear()

    # Load R
    R = np.load((R_filepath))

    # 3 ----------------------------------------------------------------------------------------------------
    # Disarray and Fractional Anisotropy estimation

    # the function estimate local disarrays and fractional anisotropy and write these values also inside R
    mtrx_of_disarrays, mtrx_of_local_fa, shape_G, R = estimate_local_disarray(R, parameters,
                                                                              ev_index=2,
                                                                              _verb=_verbose,
                                                                              _verb_deep=_deep_verbose)

    # 4a ----------------------------------------------------------------------------------------------------
    # SAVE update R (R is updated by estimate_local_disarray) in a NUMPY FILES

    # create result matrix (R) filename:
    # R_filename = 'R_' + stack_prefix + '_' + str(int(parameters['roi_xy_pix'] * parameters['px_size_xy'])) + 'um.npy'
    R_prefix = R_filename.split('.')[0]
    # R_filepath = os.path.join(base_path, process_folder, R_filename)

    # Save Results in R.npy
    np.save(R_filepath, R)
    mess_strings.append('\n> R matrix is updated and saved in: {}'.format(os.path.dirname(R_filepath)))
    mess_strings.append('> with name: {}'.format(R_filename))

    for l in mess_strings:
        print(l)
    # clear list of strings
    mess_strings.clear()

    # 4b ---------------------------------------------------------------------------------------------------
    # SAVE DISARRAY AND COMPILE RESULTS TXT
    #
    # save numpy file of both disarrays matrix (calculated with arithmetic and weighted average)
    # and get their filenames

    # [estraggo lista degli attributi della classe Mode
    # e scarto quelli che cominciano con '_' perche' saranno moduli]
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

    mess_strings.append('\n> Matrix of Disarray and Fractional Anisotropy saved in:')
    mess_strings.append('> {}'.format(os.path.join(base_path, process_folder)))
    mess_strings.append('with name: \n > {}\n > {}\n > {}\n'.format(
                                                                disarray_np_filename[Mode.ARITH],
                                                                disarray_np_filename[Mode.WEIGHT],
                                                                fa_np_filename))
    mess_strings.append('\n')

    # 5 ----------------------------------------------------------------------------------------------------
    # STATISTICS ESTIMATION, HISTOGRAMS AND SAVINGS

    # estimate statistics (see class Stat) of both disarray (arithm and weighted) and of fa
    disarray_ARITM_stats = statistics_base(mtrx_of_disarrays[Mode.ARITH], invalid_value=CONST.INV)
    disarray_WEIGHT_stats = statistics_base(mtrx_of_disarrays[Mode.WEIGHT],
                                            w=mtrx_of_local_fa,
                                            invalid_value=CONST.INV)
    fa_stats = statistics_base(mtrx_of_local_fa, invalid_value=CONST.INV)

    # compile and append statistical results strings
    s1 = compile_results_strings(mtrx_of_disarrays[Mode.ARITH], 'Disarray', disarray_ARITM_stats, 'ARITH', '%')
    s2 = compile_results_strings(mtrx_of_disarrays[Mode.WEIGHT], 'Disarray', disarray_WEIGHT_stats, 'WEIGHT', '%')
    s3 = compile_results_strings(mtrx_of_local_fa, 'Fractional Anisotropy', fa_stats)
    disarray_and_fa_results_strings = s1 + ['\n\n\n'] + s2 + ['\n\n\n'] + s3

    # update mess strings
    mess_strings = mess_strings + disarray_and_fa_results_strings

    # create results.txt filename and filepath
    txt_results_filename = 'results_disarray_by_{}_G({},{},{})_limNeig{}.txt'.format(
        R_prefix,
        int(shape_G[0]), int(shape_G[1]), int(shape_G[2]),
        int(parameters['neighbours_lim']))

    if _save_csv:
        mess_strings.append('\n> CSV files are saved in:')

        # save in csv bot disarray matrices and local_FA
        for (mtrx, np_fname) in zip([mtrx_of_disarrays[Mode.ARITH], mtrx_of_disarrays[Mode.WEIGHT], mtrx_of_local_fa],
                        [disarray_np_filename[Mode.ARITH], disarray_np_filename[Mode.WEIGHT], fa_np_filename]):

            # extract only valid values (different of INV = -1)
            values = mtrx[mtrx != CONST.INV]

            # create csv filepath and save the data
            csv_filename = np_fname.split('.')[0] + '.csv'
            csv_filepath = os.path.join(base_path, process_folder, csv_filename)
            np.savetxt(csv_filepath, values, delimiter=",", fmt='%f')
            mess_strings.append('> {}'.format(csv_filepath))

    if _save_hist:
        mess_strings.append('\n> Histogram plots are saved in:')

        # zip values, description and filename
        for (mtrx, lbl, np_fname) in zip([mtrx_of_disarrays[Mode.ARITH], mtrx_of_disarrays[Mode.WEIGHT],
                                   mtrx_of_local_fa],
                                  ['Local Disarray % (Arithmetic mean)', 'Local Disarray % (Weighted mean)',
                                   'Local Fractional Anisotropy'],
                                  [disarray_np_filename[Mode.ARITH], disarray_np_filename[Mode.WEIGHT],
                                   fa_np_filename]):

            # extract only valid values (different of INV = -1)
            values = mtrx[mtrx != CONST.INV]

            # create path name
            hist_fname = '.'.join(np_fname.split('.')[:-1]) + '.tiff'  # remove .npy and add .tiff
            hist_filepath = os.path.join(base_path, process_folder, hist_fname)

            # create histograms and save they as images
            plot_histogram(values, xlabel=lbl, ylabel='Sub-Volume occurrence', filepath=hist_filepath)
            mess_strings.append('> {}'.format(hist_filepath))

    if _save_maps:
        mess_strings.append('\n> Disarray and FA plots are saved in:')

        # disarray values normalization:
        # disarrays are normalized together to safe the little differences between ARITM and WEIGH
        # invalid values are NOT removed to preserve the matrix shape -> image
        # invalid values (if present) become the minimum values
        abs_max = np.max([mtrx_of_disarrays[Mode.ARITH].max(), mtrx_of_disarrays[Mode.WEIGHT].max()])
        abs_min = np.min([mtrx_of_disarrays[Mode.ARITH].min(), mtrx_of_disarrays[Mode.WEIGHT].min()])
        dis_norm_A = 255 * ((mtrx_of_disarrays[Mode.ARITH] - abs_min) / (abs_max - abs_min))
        dis_norm_W = 255 * ((mtrx_of_disarrays[Mode.WEIGHT] - abs_min) / (abs_max - abs_min))

        # def destination folder
        dest_folder = os.path.join(base_path, process_folder)

        # create and save frame for each data (disarrays and FA)
        for (mtrx, np_fname) in zip([dis_norm_A, dis_norm_W, mtrx_of_local_fa],
                                    [disarray_np_filename[Mode.ARITH],
                                     disarray_np_filename[Mode.WEIGHT],
                                     fa_np_filename]):

            # plot frames and save they in a sub_folder (folder_path)
            folder_path = plot_map_and_save(mtrx, np_fname, dest_folder, shape_G, shape_P)
            mess_strings.append('> {}'.format(folder_path))

    # print and write into .txt
    txt_results_filepath = os.path.join(base_path, process_folder, txt_results_filename)
    write_on_txt(mess_strings, txt_results_filepath, _print=True, mode='w')
# =============================================== END MAIN () ================================================


if __name__ == '__main__':

    # ============================================== START  BY TERMINAL ======================================
    my_parser = argparse.ArgumentParser(description='Orientation analysis - 3D Structure Tensor based')
    my_parser.add_argument('-r', '--R-path', nargs='+',
                           help='absolut path of R numpy file to read and analyze) ',
                           required=True)
    my_parser.add_argument('-p', '--parameters-filename', nargs='+',
                           help='filename of parameters.txt file (in the same folder of stack)', required=True)
    my_parser.add_argument('-v', action='store_true', default=False, dest='verbose',
                           help='print additional informations')
    my_parser.add_argument('-d', action='store_true', default=False, dest='deep_verbose',
                           help='[debug mode] - print a lot of additional data, points, values etc.')
    my_parser.add_argument('-c', action='store_true', default=True, dest='csv',
                           help='save numpy results also as CSV file')
    my_parser.add_argument('-i', action='store_true', default=True, dest='histogram',
                                                  help='save histograms of results as images')
    my_parser.add_argument('-m', action='store_true', default=True, dest='maps',
                                                  help='save maps of disarray and FA as images')

    main(my_parser)
    # ============================================== START  BY TERMINAL ======================================

    # ========= START FOR DEBUG =========
    # my_parser = argparse.ArgumentParser()
    # main(my_parser)  # empty
    # ========= START FOR DEBUG =========
