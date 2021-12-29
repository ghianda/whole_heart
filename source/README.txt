(venv_wh)

adesso l'idea è:
- st_anallysis.py -> legge parameters.txt, crea R, fa analisi orientazioni, e salva R in un file numpy

- local_disarray_on_R.py -> legge parameters, legge R, e produce:
   - matrix_of_disarray (ARITH)
   - matrix_of_disarray (WEIGHT)
   - fractional anisotropy (FA)

   e relative mappe, MIP, istogrammi ecc

- plot_vectors_on_frame è hardcoded - serve per generare frames con i quiver sopra

- st_analysis_on_all_samples itera  st_anallysis.py e (opzionale) local_disarray_on_R.py
su ttti i campioni che sono dentro la basepath in input