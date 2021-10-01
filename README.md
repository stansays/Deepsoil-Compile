# Deepsoil-Compile
This contains Python scripts for compiling and consolidating (for randomized profiles) the batch_output structure of DEEPSOIL.

Organization of files upon running (sample provided in `sample` folder):
>      |-- Deepsoil_Compile.exe
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
>         |-- output_files/ --> (program output, folder created if not found)


# Matching-Assessment
This contains Python scripts for calculating the SA_RotDnn for a suite comprising of ASC (near-field) and/or SZ (far-field) ground motion time-histories, in accordance with ASCE 7-16 Sec. 16.2.3.

Bug fixes, addition of UI, and other improvements are ongoing but this shall work given the following:
1. ASC records should have components with (H1, H2) or (FN, FP) in their respective filenames.
2. SZ records should have components with (SZ1, SZ2) in their respective filenames.

Organization of files upon running (sample provided in `sample` folder):
>      |-- Matching-Assessment.exe    
>      |-- data/
>         |-- input_files/
>             |-- 01 [Record_1]
>                |-- 01 [component].txt
>                |-- 01 [component].txt
>                ...
>             |-- 02 [Record_2]
>             ...
>             |-- {txt file with name containing "ASC" and/or "SZ", "RotD{percentile} Target", and damping % in parenthesis}
>         |-- output_files/ --> (program output, folder created if not found)
