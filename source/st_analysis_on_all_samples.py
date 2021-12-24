# run st_analysis.py on all the samples in the input basepath
# example:
# .../basepath/N1/ <- {N1.tif, parameters.txt}
# .../basepath/N2/ <- {N2.tif, parameters.txt}
# .../basepath/N2/ <- {N3.tif, parameters.txt}
# > python disarray_on_all -s /.../basepath   -> run st_analysis.py -s /.../N1; st_analysis.py -s /.../N1 etc. 

import argparse
import os
import subprocess

class Bcolors:
    VERB = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def main(parser):

    
    args = parser.parse_args()
    source_path = args.source_path[0]

    # list of subfolders
    samples = os.listdir(source_path)
    samples.sort()

    print(Bcolors.WARNING + '\n\n**************** Start Disarray Analysis on all Samples in :' + Bcolors.ENDC)
    print('- source_path : ', source_path)
    print('- list of samples:')
    for s in samples: 
        print(s)

    # run analisys on each sample
    for s in samples:
        print(Bcolors.WARNING + '\n Start the analysis on: Sample {}\n'.format(s) + Bcolors.ENDC)
        
        # sample path
        spath = os.path.join(source_path, s)

        # tiff file here
        tiff_fnames = [t for t in os.listdir(spath) if t.endswith('.tif') or t.endswith('.tiff')]
        
        # parameters file here
        par_fnames = [p for p in os.listdir(spath) if p.startswith('parameters')]
        
        # check if there is only one sample and one param file in the current directory
        if (len(tiff_fnames), len(par_fnames)) == (1, 1):
            
            # create path
            tiffpath = os.path.join(spath, tiff_fnames[0])

            # perform analsys calling the sub-script
            os.system('python3 st_analysis.py -s {} -p {}'.format(tiffpath, par_fnames[0]))
            # subprocess.call(['secondary.py', '-sf', mask_path])
    
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run st_analysys on all samples in the input path')
    parser.add_argument('-s', '--source_path', nargs='+', required=False)
    main(parser)


