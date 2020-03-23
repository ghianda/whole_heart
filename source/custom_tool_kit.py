import numpy as np
import math


# ===================================     CLASSES    ========================================


class Bcolors:
    VERB = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ===================================    METHODS    ================================================


def extract_parameters(filename, param_names, _verb=False):
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


def write_on_txt(strings, txt_path, _print=False, mode='a'):
    # write the lines in 'strings' list into .txt file addressed by txt_path
    # if _print is True, the lines is printed
    #
    with open(txt_path, mode=mode) as txt:
        for s in strings:
            txt.write(s + '\n')
            if _print:
                print(s)


def all_words_in_txt(filepath):
    words = list()
    with open(filepath, 'r') as f:
        data = f.readlines()
        for line in data:
            for word in line.split():
                words.append(word)
    return words


def search_value_in_txt(filepath, strings_to_search):
    # strings_to_search is a string or a list of string
    if type(strings_to_search) is not list:
        strings_to_search = [strings_to_search]

    # read all words in filepath
    words = all_words_in_txt(filepath)

    # search strings
    values = [words[words.index(s) + 2] for s in strings_to_search if s in words]

    return values


def manage_path_argument(source_path):
    """
    # manage input parameters:
    # if script is call by terminal, source_path is a list with one string inside (correct source path)
    # if script is call by another script (structural_analysis, for example), and if source_path contains some ' ' whitesace,
    # system split the string in the list, so it joins it

    :param source_path : variables from args.source_folder
                         it's a list with inside the path of the images to processing.
    :return
    """
    try:
        if source_path is not None:
            if type(source_path) is list:
                if len(source_path) > 1:
                    # there are white spaces, system split in more string (wrong)
                    given_path = ' '.join(source_path)
                else:
                    given_path = source_path[0]
            else:
                given_path = source_path

            # # correct whitespace with backslash
            # given_path = given_path.replace(' ', '\ ')

            # extract base path
            if given_path.endswith('/'):
                given_path = given_path[0:-1]
            return given_path
        else:
            return None

    except:
        print(Bcolors.FAIL + '[manage_path_argument] -> source_path is empty?' + Bcolors.ENDC)
        return None


def pad_dimension(matrix, shape):
    ''' check if matrix have dimension like 'shape'.
    If not, pad with zero for every axis.'''
    if matrix is not None and shape is not None:

        # check on 0th axis
        if matrix.shape[0] < shape[0]:
            zero_pad = np.zeros((int(shape[0] - matrix.shape[0]), matrix.shape[1], matrix.shape[2]))
            matrix = np.concatenate((matrix, zero_pad), 0)

        # check on 1th axis
        if matrix.shape[1] < shape[1]:
            zero_pad = np.zeros((matrix.shape[0], int(shape[1] - matrix.shape[1]), matrix.shape[2]))
            matrix = np.concatenate((matrix, zero_pad), 1)

        # check on 2th axis
        if matrix.shape[2] < shape[2]:
            zero_pad = np.zeros((matrix.shape[0], matrix.shape[1], int(shape[2] - matrix.shape[2])))
            matrix = np.concatenate((matrix, zero_pad), 2)

        return matrix

    else:
        raise ValueError('Block or shape is None')


def create_coord_by_iter(r, c, z, shape_P, _z_forced=False):
    # create init coordinate for parallelepiped

    row = r * shape_P[0]
    col = c * shape_P[1]

    if _z_forced:
        zeta = z
    else:
        zeta = z * shape_P[2]

    return (row, col, zeta)


def nextpow2(n):
    return int(np.power(2, np.ceil(np.log2(n))))


def magnitude(x):
    if x is None:
        return None
    if x < 0:
        return magnitude(abs(x))
    elif x == 0:
        return -1
    else:
        return int(math.floor(math.log10(x)))


# def normalize_m0_std(mat):
#     std = np.std(mat)
#     if std != 0:
#         # return mat
#         return ((mat - np.mean(mat)) / std).astype(np.float32)
#     else:
#         return (mat - np.mean(mat)).astype(np.float32)


def create_slice_coordinate(start_coord, shape_of_subblock):
    # create slice coordinates for take a submatrix with shape = shape_of_subblock
    # that start at start_coord
    selected_slice_coord = []
    for (start, s) in zip(start_coord, shape_of_subblock):
        selected_slice_coord.append(slice(start, start + s, 1))
    return selected_slice_coord


def seconds_to_min_sec(sec):
    # convert seconds to (hour, minutes, seconds)

    if sec < 60:
        return int(0), int(0), int(sec)
    elif sec < 3600:
        return int(0), int(sec // 60), int(sec % 60)
    else:
        h = int(sec // 3600)
        return h, int(sec // 60) - h * 60, int(sec % 60)

