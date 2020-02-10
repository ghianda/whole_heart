import os
import argparse
import numpy as np
import math
import meshIO as meshIO

import concurrent.futures
import multiprocessing


class Bcolors:
    VERB = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_thread_pool_executor(max_workers=multiprocessing.cpu_count() - 2):
    return concurrent.futures.ThreadPoolExecutor(max_workers)


def prepend_line(new_line, txt_path):
    # We read the existing lines from file in READ mode
    src = open(txt_path, 'r')
    lines = src.readlines()

    # Here, we prepend the string we want to on first line
    lines.insert(0, new_line)
    src.close()

    # We again open the file in WRITE mode to overwrite
    src = open(txt_path, 'w')
    src.writelines(lines)
    src.close()


def main(parser):

    # collect arguments
    args = parser.parse_args()
    mesh_basename = args.mesh_basename
    R_filename = args.R_filename
    _vpts = args.vpts  # bool

    print(Bcolors.OKBLUE + ' *** Filling mesh with real fibers ...' + Bcolors.ENDC)

    # current diretcory
    cwd = os.getcwd()

    # path
    mesh_basepath = os.path.join(cwd, mesh_basename)
    R_path = os.path.join(cwd, R_filename)

    # Reads in mesh pts file
    pts = meshIO.read_pts(basename=mesh_basepath, file_pts=None)
    print('- Mesh has', len(pts), 'nodes')

    # Reads in mesh elems file
    elems = meshIO.read_elems(basename=mesh_basepath, file_elem=None)
    print('- Mesh has', len(elems), 'elements')

    # Reads in mesh lon file
    lon = meshIO.read_lon(basename=mesh_basepath, file_lon=None)
    print('- Mesh has', len(lon), 'fibres')

    # Reads in mesh centroids file
    cpts = meshIO.read_cpts(basename=mesh_basepath, file_cpts=None)
    print('Mesh has', len(cpts), 'centroids')

    # Reads in vector file
    R = np.load(R_path)
    print('Successfully read {}'.format(R_filename))
    print('Fibres has shapes file (y,x,z):', R.shape)

    # ectract eigenvector components
    ev_index = 2
    fibres_YXZ = R['ev'][..., ev_index]  # quiver = fibres

    # Defines the voxel resolution of the data (ds388)
    data_ps_yxz = np.array([5.2, 5.2, 6])  # um
    print('Resolution of the data (r,c,z)', data_ps_yxz, 'um')

    # defines the grane of the orientation analysis
    # (read 'Dimension of Parallelepiped' inside Orientation_INFO.txt)
    grane_yxz = np.array([26, 26, 22])
    print('Resolution of the orientation analysis (r,c,z)', grane_yxz, 'px')

    # defines resultion of the mesh and of the segmentation
    mesh_ps_yxz = np.array([20.8, 20.8, 20])  # um
    print('Resolution of the mesh (r,c,z)', mesh_ps_yxz, 'um')

    # calculate R pixel size (the resolution of the fibres)
    R_ps = data_ps_yxz * grane_yxz
    print('Resolution of the fibres (r,c,z)', R_ps, 'um')
    print()

    # pixel size ds388
    res_x = R_ps[1]  # column
    res_y = R_ps[0]  # row
    res_z = R_ps[2]  # depth

    # Converts cpts file from um to voxels     <<<---  (doesn't round yet)
    cpts_px = cpts.copy()
    cpts_px.loc[:, 0] = cpts.loc[:, 0] / res_x  # column = x
    cpts_px.loc[:, 1] = cpts.loc[:, 1] / res_y  # row = y
    cpts_px.loc[:, 2] = cpts.loc[:, 2] / res_z  # depth = z

    # check dimension
    print(' Shape of Fibres Matrix (YXZa): ', fibres_YXZ.shape)
    print(' Max rounded coordinates of centroids (XYZ): ', np.floor(np.array(cpts.max())))

    # evaluate the integer coordinates of centroids in the px space
    cpts_int = np.ndarray((len(cpts_px), 3)).astype(np.uint16)  # (n_points, 3)
    cpts_int[:, 0] = np.floor(np.array(cpts_px.loc[:, 0]))  # x = column
    cpts_int[:, 1] = np.floor(np.array(cpts_px.loc[:, 1]))  # y = row
    cpts_int[:, 2] = np.floor(np.array(cpts_px.loc[:, 2]))  # z = depth

    # Defines a copy of the fibre file to edit
    lon_mapped = lon.copy()

    # Compile .lon with my data  --> LONG TASK (minutes...)
    # Iterates over the mesh centroids, computing the effective voxel location
    n_points = len(cpts)
    print('Start compiling the new .lon file... ')
    for i in range(n_points):

        # Sees which voxel we're in
        x = cpts_int[i, 0]  # Col = x
        y = cpts_int[i, 1]  # Row = y
        z = cpts_int[i, 2]  # Dep = z

        # insert data
        try:
            lon_mapped.loc[i][0] = fibres_YXZ[y, x, z, 1]  # X components
            lon_mapped.loc[i][1] = fibres_YXZ[y, x, z, 0]  # Y components
            lon_mapped.loc[i][2] = fibres_YXZ[y, x, z, 2]  # Z components

        except ValueError as e:
            print(Bcolors.FAIL)
            print('i={} - position: (x, y, z) = ({},{},{})'.format(i, x, y, z))
            print('FAIL with ValueError: {}'.format(e))
            print(Bcolors.ENDC)
        except:
            print(Bcolors.FAIL + 'FAIL with unknown error')
            print('i={} - position: (x, y, z) = ({},{},{})'.format(i, x, y, z))
            print(Bcolors.ENDC)

    # Writes-out mapped lon file
    filled_lon_filename = mesh_basepath + '_filled'
    print('Successfully fill the real fibers into {}.lon'.format(filled_lon_filename))
    meshIO.write_lon(lonFilename=filled_lon_filename, lon=lon_mapped)

    # creation of .vpts and .vec file to meshalyzer visualization
    if _vpts:
        print(Bcolors.OKBLUE + ' *** Creation of .vpts and .vec file for visualization...' + Bcolors.ENDC)
        # define new filenames of .vec and .vpts
        vec_filepath = mesh_basepath + '.vec'  # -> ci inserirò le componenti dei vettori (componenti)
        vpts_filepath = mesh_basepath + '.vpts'  # -> ci inserirò le coordinate dei cpts  (posizioni)

        # define the reducing factor of points and vectors
        scale = 100

        # scale the number of centroids
        scaled_cpts = cpts.loc[0:len(cpts):scale]
        print('Selected {} centroids from {} total.'.format(len(scaled_cpts), len(cpts)))

        # save selected centroid points into a .vpts file
        scaled_cpts.to_csv(vpts_filepath, sep=' ', header=False, index=False, mode='w')

        # add in the first line of .vpts file the number of selected centroids
        prepend_line(str(len(scaled_cpts)), vpts_filepath)
        print('Saved selected centroids in:\n -', vpts_filepath)

        # scale the number of vectors
        scaled_vec = lon_mapped.loc[0:len(lon_mapped):scale]
        print('Selected {} fibers vectors from {} total'.format(len(scaled_vec), len(lon_mapped)))

        # fill the new .vec array adding a new column for color plot
        # [X  Y  Z  V] : X, Y, and Z are the vector components, V the scalar for the color
        # PAY ATTENTION <------------------- now I simply copy teh z component
        scaled_vec['3'] = scaled_vec[2]

        # save new .vec file
        scaled_vec.to_csv(vec_filepath, sep=' ', header=False, index=False, mode='w')
        print('Saved selected fibres in:\n -', vec_filepath)


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Insert real fibers from into .lon file')
    my_parser.add_argument('-msh',
                           '--mesh-basename',
                           nargs='+',
                           help='absolut path of mesh basename',
                           required=True)
    my_parser.add_argument('-r',
                           '--R-filename',
                           nargs='+',
                           help='absolut path of R numpy filename',
                           required=True)
    my_parser.add_argument('-v',
                           action='store_true',
                           default=True,
                           dest='vpts',
                           help='Add \'-v\' if you want to save the new fibers also in a .vec and .vpts files '
                                'for meshalyzer visualization.')

    main(my_parser)
