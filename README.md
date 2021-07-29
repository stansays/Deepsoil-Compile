# Deepsoil-Compile
This contains Python scripts for compiling and consolidating (for randomized profiles) the batch_output structure of DEEPSOIL.

A one-file bundled executable is included in the dist folder for Windows users.


# Matching-Assessment
This contains Python scripts for calculating the SA_RotD100 and SA_GeoMean spectra for a suite comprising of ASC (near-field) and/or SZ (far-field) ground motion time-histories, in accordance with ASCE 7-16 Sec. 16.2.3.

Bug fixes, addition of UI, and other improvements are on-going but this shall work given the following:
1. Damping ratio is 5%.
2. ASC records should have components with (H1, H2) or (FN, FP) in their respective filenames.
3. SZ records should have components with (SZ1, SZ2) in their respective filenames.
4. An executable is planned to be included. Before execution, it shall be placed in a folder; this folder shall be in the same location as the ACC TH record folders.

Output files (response spectra plots and results spreadsheet) are all placed in an output folder.
