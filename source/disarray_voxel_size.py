import numpy as np


def main():
	# EXTRACT INFORMATION ABOUT PIXEL SIZE IN THE AVERAGED VOLUMES (like disarray and local_FA)
	#
	#					Insert Parameters:
	#
	# =======================================================
	# volume dimension in px
	shape_V = np.array([3192, 2316, 1578])  # px, rcz

	# array of resolutions in the R analysis
	res_array_rcz = np.array([5.2, 5.2, 6])  # um

	# dimension of the block analysis
	shape_P = np.array([26,26,22])

	# dimension of the disarray matrix
	shape_D = np.array([62,45,36])

	# dimension of analysis grane of disarray
	shape_G = np.array([2,2,2]).astype(np.uint32)

	# Insert here disarray block sides (in pixel) in the plot: 
	# (manually measured with a line in fiji)
	r_dis = 19  # px
	c_dis = 19  # px
	# =======================================================


	res_xy = res_array_rcz[1]
	res_z = res_array_rcz[2]

	# resolution of disarray estimation in um
	disarray_grane_um = res_array_rcz * shape_V / shape_D

	print('Vol: ', shape_V, '\n - disarray matrix: ', shape_D)
	print('--> disarray estimation grane in px: ',shape_P * shape_G)
	print('--> disarray estimation grane in um: ',disarray_grane_um)

	print('\nRisoluzione 3d del volume del plot del disarray (DA IMPOSTARE SU FIJI):')
	print('r, c -> ({0:0.2f}, {1:0.2f})um = (res_xy * roi_xy_pix * shape_G / block_side_in_plot)'.format(
	    res_xy * shape_P[0] * shape_G[0] / r_dis,
	    res_xy * shape_P[1] * shape_G[1] / c_dis))
	print('z  ---> {0:0.2f} um'.format(disarray_grane_um[2]))

	# print("\nPixel size in the quiver plot")
	# quiver_r_side = 1370 
	# quiver_c_side = 893
	# quiver_n_frames = 1452
	# print('r: ', res_array_rcz[0], ' --> ', res_array_rcz[0] * shape_V[0] / quiver_r_side)
	# print('c: ', res_array_rcz[1], ' --> ', res_array_rcz[1] * shape_V[1] / quiver_c_side)
	# print('z: ', res_array_rcz[2], ' --> ', res_array_rcz[2] * shape_V[2] / quiver_n_frames)

	print('\n\n')
	print("Riprova DIMENSIONI:")
	print('   shape_V       *    res_array_rcz    =    vol_in_um')
	print(shape_V, '*', res_array_rcz, '=', shape_V * res_array_rcz, 'um')
	print()
	print('   shape_D    *     disarray_grane_um     =    vol_in_um')
	print(shape_D, '*', disarray_grane_um, '=', shape_D * disarray_grane_um, 'um')
	print()
	print('shape_G * shape_D *    shape_P    = vol_enlarged_>in_px  =    vol_enlarged_in_um')
	print(shape_G, '*', shape_D, '*', shape_P, '=', 
	      shape_G * shape_D * shape_P, 'px = ',
	      shape_G * shape_D * shape_P * res_array_rcz, 'um')


if __name__ == '__main__':
	main() 
