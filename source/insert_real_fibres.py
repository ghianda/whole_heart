import os
import argparse
import numpy as np
import math
import source.meshIO as meshIO

from source.custom_tool_kit import Bcolors


def main(parser):

    # collect arguments
    args = argparse.parser()
    mesh_basename = args.mesh_basename
    R_filename = args.R_filename

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
    cpts.loc[:, 0] = cpts.loc[:, 0] / res_x  # column = x
    cpts.loc[:, 1] = cpts.loc[:, 1] / res_y  # row = y
    cpts.loc[:, 2] = cpts.loc[:, 2] / res_z  # depth = z

    # Defines a copy of the fibre file to edit
    lon_mapped = lon.copy()
    lon

    # check dimension
    print(' Shape of Fibres Matrix (YXZa): ', fibres_YXZ.shape)
    print(' Max rounded coordinates of centroids (XYZ): ', np.floor(np.array(cpts.max())))

    # evaluate the integer coordinates of centroids in the px space
    cpts_int = np.ndarray((len(cpts), 3)).astype(np.uint16)  # (n_points, 3)
    cpts_int[:, 0] = np.floor(np.array(cpts.loc[:, 0]))  # x = column
    cpts_int[:, 1] = np.floor(np.array(cpts.loc[:, 1]))  # y = row
    cpts_int[:, 2] = np.floor(np.array(cpts.loc[:, 2]))  # z = depth

    # Compile .lon with my data  --> LONG TASK (minutes...)
    # Iterates over the mesh centroids, computing the effective voxel location
    n_points = len(cpts)
    magn = math.floor(math.log10(n_points))
    for i in range(n_points):

        # print percentual
        if i % 10**(magn - 1) == 0:
            print(' - {}%'.format(100 * i / n_points))

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
