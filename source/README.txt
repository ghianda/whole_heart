(venv_wh)


> st_analysis.py -s path_tiff -p parameters_filename.txt

fa l'analisi completa, orientazinoni, disarray, salvataggi ecc


> disarray_on_R.py -r R_filepath -p parameters_filename

bypassa l'analisi delle orientazioni e fa il disarray leggendo dalla matrice R in R_filepath


TODO:
> funz che salva il disarray come mappe tiff:
 --- senza normalizzare
 --- con la pixel size giusta
 (esempio: 1 pixel per ogni elemento della matrice disarray -
 tanto la pixel size del disarray la so facendo ps * grana_orient * grana_disarray)