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
import sys
import argparse

# general
import numpy as np
from numpy import linalg as LA
from zetastitcher import InputFile

# image
from skimage.filters import gaussian
from skimage import transform
from scipy import ndimage as ndi

# custom codes
from custom_tool_kit import manage_path_argument, create_coord_by_iter, create_slice_coordinate, \
    all_words_in_txt, search_value_in_txt, pad_dimension, write_on_txt
from custom_image_base_tool import normalize, print_info
from local_disarray_by_R import estimate_local_disarry


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Cell_Ratio_mode:
    NON_ZERO_RATIO = 0.0
    MEAN = 1.0


def create_R(shape_V, shape_P):
    '''
    Define Results Matrix - 'R'

    :param shape_V: shape of entire Volume of data
    :param shape_P: shape of parallelepipeds extracted for orientation analysis
    :return: empty Result matrix 'R'
    '''

    if any(shape_V > shape_P):
        # shape
        shape_R = np.ceil(shape_V / shape_P).astype(np.int)

        # define empty Results matrix
        total_num_of_cells = np.prod(shape_R)
        R = np.zeros(
            total_num_of_cells,
            dtype=[('id_block', np.int64),  # unique identifier of block
                   ('cell_info', bool),  # 1 if block is analyzed, 0 if it is rejected by cell_threshold
                   ('orient_info', bool),  # 1 if block is analyzed, 0 if it is rejected by cell_threshold
                   ('cell_ratio', np.float16),  # ratio between cell voxel and all voxel of block
                   ('init_coord', np.int32, (1, 3)),  # absolute coord of voxel block[0,0,0] in Volume
                   ('ew', np.float32, (1, 3)),  # descending ordered eigenvalues.
                   ('ev', np.float32, (3, 3)),  # column ev[:,i] is the eigenvector of the eigenvalue w[i].
                   ('strenght', np.float16),  # parametro forza del gradiente (w1 .=. w2 .=. w3)
                   ('cilindrical_dim', np.float16),  # dimensionalità forma cilindrica (w1 .=. w2 >> w3)
                   ('planar_dim', np.float16),  # dimensionalità forma planare (w1 >> w2 .=. w3)
                   ('fa', np.float16), # fractional anisotropy (0-> isotropic, 1-> max anisotropy
                   ('local_disarray', np.float16),  # where store local_disarray results
                   ]
        ).reshape(shape_R)

        # initialize mask of info to False
        R[:, :, :]['cell_info'] = False
        R[:, :, :]['orient_info'] = False

        print('Dimension of Result Matrix:', R.shape)
        return R, shape_R
    else:
        raise ValueError(' Data array dimension is smaller than dimension of one parallelepiped. \n'
                         ' Ensure which data is loaded or modify analysis parameter')


def check_in_upper_semisphere(v):
    # if versor in in y < 0 semisphere, take versor with opposite direction
    if v[1] >= 0:
        v_rot = np.copy(v)
    else:
        v_rot = -np.copy(v)
    return v_rot


def sigma_for_uniform_resolution(FWHM_xy, FWHM_z, px_size_xy):
    # Legend: SD: Standard Deviation = sigma; Var: Variance = sigma**2
    #
    # For uniform resolution over 3 axes, xy plane are blurred by a gaussian kernel with sigma = sigma_s.
    # Sigma_s depends by resolution (FWHM) in xy and Z
    # f_xy convoled f_s = f_z, where f is Gaussian profile of 1d PSF.
    # so, sigma_xy**2 + sigma_s**2 = sigma_z**2
    #
    # That function calculate sigma_s (in micron) by converting FWHM_xy and FWHM_z in sigma_xy and sigma_z,
    # and calculate sigma_s in pixel by pixel sixe in xy.

    # estimate variance (sigma**2) by FWHM (in micron)
    sigma2_xy = FWHM_xy ** 2 / (8 * np.log(2))
    sigma2_z = FWHM_z ** 2 / (8 * np.log(2))

    # estimate variance (sigma_s**2) of gaussian kernel (in micron)
    sigma2_s = np.abs(sigma2_z - sigma2_xy)

    # estimate SD of Gaussian Kernel (in micron)
    sigma_s = np.sqrt(sigma2_s)

    # return SD in pixel
    return sigma_s / px_size_xy


def downsample_2_zeta_resolution(vol, px_size_xy, px_size_z, sigma):
    # Opera uno smoothing in xy per portare la larghezza della PSF in xy UGUALE alla larghezza della PSF in Zeta.
    # Poi fa downsampling per uniformare anche la pixel size.
    # L'immagine prodotta avrà le stesse caratteristiche delle sezioni XZ e YZ,
    # in modo da rendere il volume a risoluzione isotropa e
    # l'operazione di derivazione nei tre assi priva di sbilanciamenti.
    #
    # vol is np.uint8
    # downsampled is np.float32

    # estimate new isotropic sizes
    resize_ratio = px_size_xy / px_size_z
    resized_dim = int(resize_ratio * vol.shape[0])

    downsampled = np.zeros((resized_dim, resized_dim, vol.shape[2]), dtype=np.float32)

    for z in range(vol.shape[2]):
        # convolve with gaussian kernel for smooth
        blurred = gaussian(image=vol[:, :, z], sigma=sigma, mode='reflect')

        # resize and save into new downsampled volume
        downsampled[:, :, z] = normalize(
            img=transform.resize(image=blurred, output_shape=(resized_dim, resized_dim), mode='reflect'),
            dtype=np.float32)

    return downsampled


def Structure_Tensor_Analysis_3D(vol):
    # TODO DA MMODIFICARE COMMENTI (farlo per 3d)

    #
    # Structure Tensor definita così:
    # ogni elemento (i,j) dello ST sarà la media della matrice IiIj
    # create ST matrix with mean of Ixx, Ixy and Iyy
    #
    # |IxIx  IxIy  IxIz|          |mean(IxIx)  mean(IxIy)  mean(IxIz)|
    # |IyIx  IyIy  IyIz|     ->   |mean(IyIx)  mean(IyIy)  mean(IyIz)|
    # |IzIx  IzIy  IzIz|     ->   |mean(IzIx)  mean(IzIy)  mean(IzIz)|
    #   (3 x 3 x m x m)      -->                 (3 x 3)
    #    m : ROI side
    #
    #
    # INPUT: vol (rcz = yxz)
    # OUTPUT: (eigenvalues, eigenvectors, shape_parameters)
    # eigenvalues and eigenvectors are oredered by descending eigenvalues values.
    # w[0] > w[1] >...

    # Compute Gradient over x y z directions
    gx, gy, gz = np.gradient(vol)

    # compute second order moment of gradient
    Ixx = ndi.gaussian_filter(gx * gx, sigma=1, mode='constant', cval=0)
    Ixy = ndi.gaussian_filter(gx * gy, sigma=1, mode='constant', cval=0)
    Ixz = ndi.gaussian_filter(gx * gz, sigma=1, mode='constant', cval=0)
    # Iyx = Ixy, simmetric
    Iyy = ndi.gaussian_filter(gy * gy, sigma=1, mode='constant', cval=0)
    Iyz = ndi.gaussian_filter(gy * gz, sigma=1, mode='constant', cval=0)
    # Izx = Ixz, simmetric
    # Izy = Iyz, simmetric
    Izz = ndi.gaussian_filter(gz * gz, sigma=1, mode='constant', cval=0)

    # create ST matrix with mean of Ixx, Ixy and Iyy
    # |IxIx  IxIy  IxIz|          |mean(IxIx)  mean(IxIy)  mean(IxIz)|
    # |IyIx  IyIy  IyIz|     ->   |mean(IyIx)  mean(IyIy)  mean(IyIz)|
    # |IzIx  IzIy  IzIz|     ->   |mean(IzIx)  mean(IzIy)  mean(IzIz)|
    #   (3 x 3 x m x m)      -->                 (3 x 3)
    ST = np.array([[np.mean(Ixx), np.mean(Ixy), np.mean(Ixz)],
                   [np.mean(Ixy), np.mean(Iyy), np.mean(Iyz)],
                   [np.mean(Ixz), np.mean(Iyz), np.mean(Izz)]])

    # eigenvalues and eigenvectors decomposition
    w, v = LA.eig(ST)
    # NB : the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].

    # ordino autovalori e autovettori t.c. w0 > w1 > w2
    order = np.argsort(w)[::-1]  # decrescent order
    w = np.copy(w[order])
    v = np.copy(v[:, order])

    # sposta autovettori sulla semisfera con y > 0
    ev_rotated = np.zeros_like(v)
    for axis in range(v.shape[1]):
        ev_rotated[:, axis] = check_in_upper_semisphere(v[:, axis])

    # parameri di forma
    shape_parameters = dict()

    # calcolo fractional anisotropy (0 isotropy -> 1 anisotropy)
    shape_parameters['fa'] = np.sqrt(1/2) * (
        np.sqrt((w[0] - w[1]) ** 2 + (w[1] - w[2]) ** 2 + (w[2] - w[1]) ** 2) / np.sqrt(np.sum(w ** 2))
    )

    # calcolo parametro forza del gradiente (per distinguere contenuto da sfondo)
    # (w1 .=. w2 .=. w3)
    shape_parameters['strenght'] = np.sqrt(np.sum(w))

    # calcolo dimensionalità forma cilindrica (w1 .=. w2 >> w3)
    shape_parameters['cilindrical_dim'] = (w[1] - w[2]) / (w[1] + w[2])

    # calcolo dimensionalità forma planare (w1 >> w2 .=. w3)
    shape_parameters['planar_dim'] = (w[0] - w[1]) / (w[0] + w[1])

    return (w, ev_rotated, shape_parameters)


def block_analysis(parall, shape_P, parameters, sigma, _verbose):
    # parall : np.uint8

    # initialize empty dictionary and
    results = {}
    there_is_cell = there_is_info = False

    # check if this block contains cell with the selected methods:
    if parameters['mode_ratio'] == Cell_Ratio_mode.MEAN:
        cell_ratio = np.mean(parall)
    elif parameters['mode_ratio'] == Cell_Ratio_mode.NON_ZERO_RATIO:
        cell_ratio = np.count_nonzero(parall) / np.prod(shape_P)
    else:
        print(Bcolors.WARNING + '** WARNING: parameters[\'mode_ratio\'] is not recognized: all blacks are not analyzed' + Bcolors.ENDC)
        cell_ratio = 0

    if cell_ratio > parameters['threshold_on_cell_ratio']:
        # Orientation Analysis in this data block
        there_is_cell = True

        # save in R
        results['cell_ratio'] = cell_ratio
        if _verbose: print('   cell_ratio :   ', cell_ratio)

        # blurring (for isotropic FWHM) and downsampling (for isotropic pixel size)
        # parall is int_8, parall_down is float_32
        parall_down = downsample_2_zeta_resolution(parall,
                                                   parameters['px_size_xy'],
                                                   parameters['px_size_z'],
                                                   sigma=sigma)
        # print('parall_down mean: {}'.format(np.mean(parall_down)))

        # 3D Structure Tensor Analysis - Gradient based
        # - w : descendent ordered eigenvalues
        # - v : ordered eigenvectors
        #       the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        # - shape_parameters : dictionary con form parameters
        w, v, shape_parameters = Structure_Tensor_Analysis_3D(parall_down)

        # TODO CONTROLLO SUI PARAMETRI  DI FORMA
        if shape_parameters['strenght'] > 0:
            there_is_info = True

            # save ordered eigenvectors
            results['ev'] = v
            results['ew'] = w

            # save shape parameters
            for key in shape_parameters.keys():
                results[key] = shape_parameters[key]

        else:
            if _verbose: print('Block rejected ( no info in freq )')
    else:
        if _verbose: print('Block rejected ( no cell )')

    return there_is_cell, there_is_info, results


# =================================================== MAIN () ================================================
def main(parser):

    args = parser.parse_args()

    # Extract input information
    source_path = manage_path_argument(args.source_path)
    parameter_filename = args.parameters_filename[0]

    # extract filenames and folders
    stack_name = os.path.basename(source_path)
    process_folder = os.path.basename(os.path.dirname(source_path))
    base_path = os.path.dirname(os.path.dirname(source_path))
    parameter_filepath = os.path.join(base_path, process_folder, parameter_filename)
    stack_prefix = stack_name.split('.')[0]

    # extract other preferences
    _verbose = args.verbose
    _save_csv = args.csv

    # create sointroductiveme informations
    mess_strings = list()
    mess_strings.append('\n\n*** ST orientation Analysis ***\n')
    mess_strings.append(' > source path: {}'.format(source_path))
    mess_strings.append(' > stack name: {}'.format(stack_name))
    mess_strings.append(' > process folder: {}'.format(process_folder))
    mess_strings.append(' > base path: {}'.format(base_path))
    mess_strings.append(' > Parameter filename: {}'.format(parameter_filename))
    mess_strings.append(' > Parameter filepath: {}'.format(parameter_filepath))
    mess_strings.append('')

    # TODO here added local_disarray_z_side and local_disarray_xy_side
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
    num_of_slices_P = int(parameters['roi_xy_pix'] / ps_ratio)

    row_P = col_P = int(parameters['roi_xy_pix'])
    shape_P = np.array((row_P, col_P, num_of_slices_P)).astype(np.int32)

    mess_strings.append('\n *** Analysis configuration')
    mess_strings.append(' > Rapporto fra Pixel Size (z / xy) = {0:0.2f}'.format(ps_ratio))
    mess_strings.append(' > Numero di slice selezionate per ogni ROI ({} x {}): {}'.format(
        row_P, col_P, num_of_slices_P))
    mess_strings.append(' > Dimension of Parallelepiped: ({0},{1},{2}) pixel  ='
                        '  [{3:2.2f} {4:2.2f} {5:2.2f}] um'.format(
        shape_P[0], shape_P[1], shape_P[2],
        row_P * parameters['px_size_xy'],
        col_P * parameters['px_size_xy'],
        num_of_slices_P * parameters['px_size_z']))

    # create result.txt filename:
    txt_filename = 'Orientations_' + stack_prefix + '_' \
                   + str(int(parameters['roi_xy_pix'] * parameters['px_size_xy'])) + 'um.txt'
    txt_path = os.path.join(os.path.dirname(source_path), txt_filename)

    # print and write into .txt introductive informations
    write_on_txt(mess_strings, txt_path, _print=True)
    # clear list of strings
    mess_strings.clear()

    # 1 ----------------------------------------------------------------------------------------------------
    # OPEN STACK

    # extract data - entire Volume: 'V'
    volume = InputFile(source_path).whole()
    # NB - in futuro va cambiata gestion assi
    volume = np.moveaxis(volume, 0, -1)  # (r, c, z) -> (z, y, x)

    # calculate dimension
    shape_V = np.array(volume.shape)
    pixel_for_slice = shape_V[0] * shape_V[1]
    total_voxel_V = pixel_for_slice * shape_V[2]

    mess_strings.append('\n\n*** Entire loaded Volume dimension:')
    mess_strings.append(' > Dimension if entire Volume : ({}, {}, {})'.format(shape_V[0], shape_V[1], shape_V[2]))
    mess_strings.append(' > Pixel for slice            : {}'.format(pixel_for_slice))
    mess_strings.append(' > Total voxel in Volume      : {}'.format(total_voxel_V))

    # extract list of math informations (as strings) about volume.npy variable
    info = print_info(volume, text='\nVolume informations:', _std=False, _return=True)
    mess_strings = mess_strings + info

    # print and write into .txt
    write_on_txt(mess_strings, txt_path, _print=True)
    # clear list of strings
    mess_strings.clear()

    # 2 ----------------------------------------------------------------------------------------------------
    # CYCLE FOR BLOCKS EXTRACTION and ANALYSIS
    print('\n\n')
    print('*** Start Structure Tensor analysis... ')

    t_start = time.time()

    # create empty Result matrix
    R, shape_R = create_R(shape_V, shape_P)

    # estimate sigma of blurring for isotropic resolution
    sigma_blur = sigma_for_uniform_resolution(FWHM_xy=parameters['fwhm_xy'],
                                              FWHM_z=parameters['fwhm_z'],
                                              px_size_xy=parameters['px_size_xy'])
    perc = 0
    count = 0  # count iteration
    tot = np.prod(shape_R)
    print(' > Expected iterations : ', tot)

    for z in range(shape_R[2]):
        if _verbose: print('\n\n')
        print('{0:0.1f} % - z: {1:3}'.format(perc, z))
        for r in range(shape_R[0]):
            for c in range(shape_R[1]):

                start_coord = create_coord_by_iter(r, c, z, shape_P)
                slice_coord = create_slice_coordinate(start_coord, shape_P)

                perc = 100 * (count / tot)
                if _verbose: print('\n')

                # save init info in R
                R[r, c, z]['id_block'] = count
                R[r, c, z]['init_coord'] = start_coord

                # extract parallelepiped
                parall = volume[slice_coord]

                # check dimension (if iteration is on border of volume, add zero_pad)
                parall = pad_dimension(parall, shape_P)

                # If it's not all black...
                if np.max(parall) != 0:

                    # analysis of parallelepiped extracted
                    there_is_cell, there_is_info, results = block_analysis(
                        parall,
                        shape_P,
                        parameters,
                        sigma_blur,
                        _verbose)

                    # save info in R[r, c, z]
                    if there_is_cell: R[r, c, z]['cell_info'] = True
                    if there_is_info: R[r, c, z]['orient_info'] = True

                    # save results in R
                    if _verbose: print(' saved in R:  ')
                    for key in results.keys():
                        R[r, c, z][key] = results[key]
                        if _verbose:
                            print(' > {} : {}'.format(key, R[r, c, z][key]))

                else:
                    if _verbose: print('   block rejected   ')
                    print()

                count += 1

    block_with_cell = np.count_nonzero(R['cell_info'])
    block_with_info = np.count_nonzero(R['orient_info'])
    p_rejec_cell = 100 * (1 - (block_with_cell / count))
    p_rejec_info_tot = 100 * (1 - (block_with_info / count))
    p_rejec_info = 100 * (1 - (block_with_info / block_with_cell))

    t_process = time.time() - t_start

    mess_strings.append('\n\n*** Results of Orientation analysis:')
    mess_strings.append(' > Expected iterations : {}'.format(np.prod(shape_R)))
    mess_strings.append(' > total_ iteration : {}'.format(count))
    mess_strings.append(' > Time elapsed: {0:.3f} s'.format(t_process))
    mess_strings.append('\n > Total blocks: {}'.format(count))
    mess_strings.append(' > block with cell : {0}, rejected from total: {1} ({2:0.1f}%)'.format(
        block_with_cell,
        count - block_with_cell,
        p_rejec_cell))
    mess_strings.append(' > block with gradient information : {}'.format(block_with_info))
    mess_strings.append(' > rejected from total: {0} ({1:0.1f}%)'.format(count - block_with_info, p_rejec_info_tot))
    mess_strings.append(' > rejected from block with cell: {0} ({1:0.1f}%)'.format(
        block_with_cell - block_with_info, p_rejec_info))

    # print and write into .txt
    write_on_txt(mess_strings, txt_path, _print=True)
    # clear list of strings
    mess_strings.clear()

    # 3 ----------------------------------------------------------------------------------------------------
    # Disarray estimation

    # the function estimate local disarrays and write these values also inside R
    matrix_of_disarrays, shape_G, R = estimate_local_disarry(R, parameters, ev_index=2, _verb=True, _verb_deep=False)

    # extract only valid disarray values
    disarray_values = matrix_of_disarrays[matrix_of_disarrays != -1]

    # 4 ----------------------------------------------------------------------------------------------------
    # WRITE RESULTS AND SAVE

    # create result matrix (R) filename:
    R_filename = 'R_' + stack_prefix + '_' + str(int(parameters['roi_xy_pix'] * parameters['px_size_xy'])) + 'um.npy'
    R_prefix = R_filename.split('.')[0]
    R_filepath = os.path.join(base_path, process_folder, R_filename)

    # Save Results in R.npy
    np.save(R_filepath, R)
    mess_strings.append('\n > R matrix saved in: {}'.format(os.path.dirname(source_path)))
    mess_strings.append(' > with name: {}'.format(R_filename))

    mess_strings.append('\n > Results .txt file saved in: {}'.format(os.path.dirname(txt_path)))
    mess_strings.append(' > with name: {}'.format(txt_filename))

    # create filename of numpy.file where save disarray matrix
    disarray_numpy_filename = 'MatrixDisarray_{}_G({},{},{})_limNeig{}.npy'.format(
        R_prefix,
        int(shape_G[0]), int(shape_G[1]), int(shape_G[2]),
        int(parameters['neighbours_lim']))

    mess_strings.append('\n> Matrix of Disarray saved in:')
    mess_strings.append(os.path.join(base_path, process_folder))
    mess_strings.append(' > with name: \n{}'.format(disarray_numpy_filename))

    # save numpy file
    np.save(os.path.join(base_path, process_folder, disarray_numpy_filename), matrix_of_disarrays)



    # create results strings
    mess_strings.append('\n\n*** Results of statistical analysis of Disarray on accepted points. \n')
    mess_strings.append('> Disarray (%):= 100 * (1 - alignment)\n')
    mess_strings.append('> Matrix of disarray shape: {}'.format(matrix_of_disarrays.shape))
    mess_strings.append('> Valid disarray values: {}'.format(disarray_values.shape))
    mess_strings.append('\n> Disarray mean: {0:0.2f}%'.format(np.mean(disarray_values)))
    mess_strings.append('> Disarray std: {0:0.2f}% '.format(np.std(disarray_values)))
    mess_strings.append(
        '> Disarray (min, MAX)%: ({0:0.2f}, {1:0.2f})'.format(np.min(disarray_values), np.max(disarray_values)))

    # create results.txt filename and filepath
    disarray_results_filename = 'results_disarray_by_{}_G({},{},{})_limNeig{}.txt'.format(
        R_prefix,
        int(shape_G[0]), int(shape_G[1]), int(shape_G[2]),
        int(parameters['neighbours_lim']))

    disarray_txt_filepath = os.path.join(base_path, process_folder, disarray_results_filename)

    if _save_csv:
        disarray_csv_filename = disarray_results_filename.split('.')[0] + '.csv'
        np.savetxt(os.path.join(base_path, process_folder, disarray_csv_filename), disarray_values, delimiter=",", fmt='%f')

    # print and write into .txt
    write_on_txt(mess_strings, disarray_txt_filepath, _print=True)

# =============================================== END MAIN () ================================================


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Orientation analysis - 3D Structure Tensor based')

    my_parser.add_argument('-s', '--source-path', nargs='+',
                           help='absolut path of sample to analyze (3d tiff file or folder of tiff files) ',
                           required=True)
    my_parser.add_argument('-p', '--parameters-filename', nargs='+',
                           help='filename of parameters.txt file (in the same folder of stack)', required=True)
    my_parser.add_argument('-v', action='store_true', default=False, dest='verbose',
                           help='print additional informations')

    my_parser.add_argument('-c', action='store_true', default=False, dest='csv',
                           help='save numpy results also as CSV file')

    main(my_parser)
