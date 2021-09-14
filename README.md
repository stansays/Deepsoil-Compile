# Deepsoil-Compile
This contains Python scripts for compiling and consolidating (for randomized profiles) the batch_output structure of DEEPSOIL.

A one-file bundled executable is included in the dist folder for Windows users.

>      |-- dist/
>         |-- Deepsoil_Compile.exe
>      |-- data/
>         |-- input_files/
>             |-- profile_1
>                |-- Motion_01 [component] ([damping]%)
>                   |-- deepsoilout.db3
>                |-- Motion_01 [component] ([damping]%)
>                ...
>             |-- profile_2
>                |-- Motion_02 [component] ([damping]%)
>                |-- Motion_02 [component] ([damping]%)
>             ...
>         |-- output_files/ --> (program output)


# Matching-Assessment
This contains Python scripts for calculating the SA_RotD100 and SA_GeoMean spectra for a suite comprising of ASC (near-field) and/or SZ (far-field) ground motion time-histories, in accordance with ASCE 7-16 Sec. 16.2.3.

Bug fixes, addition of UI, and other improvements are ongoing but this shall work given the following:
1. Damping ratio is 5% (UI will be provided soon).
2. ASC records should have components with (H1, H2) or (FN, FP) in their respective filenames.
3. SZ records should have components with (SZ1, SZ2) in their respective filenames.
    
>      |-- data/
>         |-- input_files/
>             |-- 01 [Record_1]
>                |-- 01 [component].txt
>                |-- 01 [component].txt
>                ...
>             |-- 02 [Record_2]
>                |-- 02 [component].txt
>                |-- 02 [component].txt
>                ...
>             ...
>             |-- [(ASC)...(Target)...]*.txt* and/or [(SZ)...(Target)...]*.txt*
>         |-- output_files/ --> (program output)
>      |-- Matching-Assessment.exe
    
4. Output files (response spectra plots and results spreadsheet) are all placed in the `output_files` folder.
