#!/usr/bin/env python3

# AMH Philippines Inc.
# FJTBernales, SBRSayson (2021.09.17)

'''
Combines scaled or spectrally-matched horizontal-component time histories and
compares with a target spectra.
'''

import os
import warnings
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.stats.mstats import gmean
from itertools import repeat
from re import search

import response_spectrum as rsp
import intensity_measures as ims
from sm_utils import convert_accel_units as conv

suite_list = []

def _import_time_series(folder_acc_data):
    """
    Import Matched Acceleration Time-Series Data given directory containing the
    record set.
    """
    try:
        os.path.isdir(folder_acc_data)
    except OSError:
        print("Folder of acceleration time series records does not exist.")

    H1_ALIASES = ['FN', 'Normal', 'H1', 'Hor1', 'SZ1']
    H2_ALIASES = ['FP', 'Parallel', 'H2', 'Hor2', 'SZ2']

    component_list = os.listdir(folder_acc_data)

    # raise warning if file prefix is not the same with dir prefix
    # inefficient search; improve with regex; check for multiple matches
    for alias in H1_ALIASES:
        for comp in component_list:
            if alias in comp:
                H1_file = comp

    for alias in H2_ALIASES:
        for comp in component_list:
            if alias in comp:
                H2_file = comp

    H1_record = pd.read_csv(os.path.join(folder_acc_data, H1_file), delim_whitespace=True)
    H2_record = pd.read_csv(os.path.join(folder_acc_data, H2_file), delim_whitespace=True)

    # check for missing fields and raise error

    H1_time = H1_record.iloc[:, 0].to_numpy()
    H2_time = H2_record.iloc[:, 0].to_numpy()

    H1_acc = H1_record.iloc[:, 1].to_numpy()
    H2_acc = H2_record.iloc[:, 1].to_numpy()

    # Check record length
    if H1_time.shape[0] != H2_time.shape[0]:
        warnings.warn(  f"Record pair in {os.path.abspath(folder_acc_data)} do "
                        f"not have the same length!")

    # Check time step
    time_step = np.diff(H1_time)[0]
    if not np.all(np.fabs(np.diff(H1_time) - time_step) < 1E-10):
        raise ValueError(f"Time step in {os.path.abspath(H1_file)} \
                            is not consistent!"                                                                                              )

    if not np.all(np.fabs(np.diff(H2_time) - time_step) < 1E-10):
        raise ValueError(f"Time step in {os.path.abspath(H2_file)} \
                            is not consistent!"                                                                                              )

    if np.diff(H1_time)[0] != np.diff(H2_time)[0]:
        raise ValueError(   f"Record pair {os.path.abspath(folder_acc_data)} "
                            f"must have the same time step!")

    return H1_acc, H2_acc, time_step


def import_ASC_target_spectra(suite_dir):
    """
    Imports ASC Target Response Spectra from a file with words 'Target' and
    'ASC' within suite directory.

    Returns dict containing periods and spectral acceleration.
    """
    try:
        os.path.isdir(suite_dir)
    except OSError:
        print("Folder of ASC target spectra does not exist.")

    component_list = os.listdir(suite_dir)

    targets = [comp for comp in component_list if 'target' in comp.lower()]
    if not targets:
        raise IndexError("Provide target spectra files!")

    # Search for ASC target
    targ_file = [targ for targ in targets if 'ASC' in targ.upper()]
    if len(targ_file) > 1: # more than 1 match
        raise ValueError("Multiple ASC target spectra found. Please check!")
    elif not any(targ_file): # if empty
        ASC_target = {}
        # check if SZ is instead specified
        alt_check = [targ for targ in targets if 'SZ' in targ.upper()]
        if not alt_check:
            raise IndexError("Specify target spectra files as ASC or SZ!")
    else:
        [targ_file] =  targ_file # unpack only member
        df = pd.read_csv(os.path.join(suite_dir, targ_file), delim_whitespace=True)

        # check for missing fields and raise error

        ASC_target = {}
        ASC_target['Periods'] = df.iloc[:, 0].to_numpy()
        ASC_target['SA'] = df.iloc[:, 1].to_numpy()

        print("Import ASC target spectra successful!")

    return ASC_target


def import_SZ_target_spectra(suite_dir):
    """
    Imports SZ Target Response Spectra from a file with words 'Target' and
    'SZ' within suite directory.

    Returns dict containing periods and spectral acceleration.
    """
    try:
        os.path.isdir(suite_dir)
    except OSError:
        raise OSError("Folder of SZ target spectra does not exist.")

    component_list = os.listdir(suite_dir)

    targets = [comp for comp in component_list if 'target' in comp.lower()]
    if not targets:
        raise IndexError("Provide target spectra files!")

    # Search for SZ target
    targ_file = [targ for targ in targets if 'SZ' in targ.upper()]
    if len(targ_file) > 1: # more than 1 match
        raise ValueError("Multiple SZ target spectra found. Please check!")
    elif not any(targ_file): # if empty
        SZ_target = {}
        # check if ASC is instead specified
        alt_check = [targ for targ in targets if 'ASC' in targ.upper()]
        if not alt_check:
            raise IndexError("Specify target spectra files as ASC or SZ!")
    else:
        [targ_file] =  targ_file # unpack only member
        df = pd.read_csv(os.path.join(suite_dir, targ_file), delim_whitespace=True)

        # check for missing fields and raise error

        SZ_target = {}
        SZ_target['Periods'] = df.iloc[:, 0].to_numpy()
        SZ_target['SA'] = df.iloc[:, 1].to_numpy()

        print("Import SZ target spectra successful!")

    return SZ_target

def detect_damping_ratio(suite_dir):
    '''
    Extracts information on Damping from target spectra files
    '''
    try:
        os.path.isdir(suite_dir)
    except OSError:
        raise OSError('Folder of target spectra does not exist.')

    component_list = os.listdir(suite_dir)

    targets = [comp for comp in component_list if 'target' in comp.lower()]
    if not targets:
        raise IndexError('Provide target spectra files!')

    damping = []
    for target in targets:
        damping.append(
            float(target[target.find('(') + 1:target.find('%')]) / 100)
    if len(set(damping)) != 1:
        raise ValueError('Damping ratios in target files are inconsistent!')
    else:
        print(f"Damping = {damping[0] * 100:.2f}%")
        return damping[0]

def detect_percentile(suite_dir):
    '''
    Extracts information on percentile from target spectra files
    '''
    try:
        os.path.isdir(suite_dir)
    except OSError:
        raise OSError('Folder of target spectra does not exist.')

    component_list = os.listdir(suite_dir)

    targets = [comp for comp in component_list if 'target' in comp.lower()]
    if not targets:
        raise IndexError('Provide target spectra files!')
    
    percentile = []
    for target in targets:
        percentile.append(float(target[target.lower().find('rotd') + 4:
                        target.lower().find('target')]))
    if len(set(percentile)) != 1:
        raise ValueError('Percentile in target files are inconsistent!')
    else:
        print(f"Percentile = {percentile[0]:.0f}%")
        return percentile[0]

def _record_rotdnn(record, suite_dir, ALIASES, periods, damping_level,
                    percentile, suite_rotdnn):
    '''
    function for parallel processing of records in suite list
    '''
    record_path = os.path.join(suite_dir, record)
    # this can be removed once repetitive check is fixed
    component_list = os.listdir(record_path)

    matching_alias = [alias in comp for alias in ALIASES \
                        for comp in component_list]

    if not any(matching_alias):
        # skip RotDnn computation if no ASC/SZ file names found in record dir
        return

    elif sum(matching_alias) == 1:
        # Raise error message if only 1 matching name is found.
        raise ValueError(   f"No matching record pair found in "
                            f"{os.path.abspath(record)}.")
    # import corresponding time series from record set dir
    record_acc_H1, record_acc_H2, dt = _import_time_series(record_path)

    # calculate RotDnn output dict for record, but result is in cgs units
    RotDnn_cgs = ims.rotdpp(record_acc_H1, dt, record_acc_H2, dt,
                            periods, percentile=percentile,
                            damping=damping_level, units='g')[0]
    # # FOR WORKFLOW TESTING ONLY
    # sax, say = ims.get_response_spectrum_pair(
    #                 record_acc_H1, dt, record_acc_H2, dt,
    #                 periods, damping=damping_level, units='g')
    #
    # # calculate GeoMean spectra for record pair, but result is in cgs units
    # RotDnn_cgs = ims.geometric_mean_spectrum(sax, say)

    # convert to g units
    RotDnn_SA_g = conv(RotDnn_cgs["Pseudo-Acceleration"],
                        from_='cm/s/s', to_='g')
    # append to suite dict
    suite_rotdnn[record] = RotDnn_SA_g

def compute_suite_rotdnn_spectra(suite_dir, periods, trt,
                                percentile, damping_level=0.05):
    """
    Calculate RotDnn Response Spectra for each record in a given suite of
    acceleration time-histories from ASC/SZ regions.
    """
    print("Calculting " + trt + f" RotD{percentile:.0f}-component spectra...")

    dir_list = os.listdir(suite_dir)
    manager = mp.Manager()

    # raise warning for duplicate prefixes?
    # raise warning if file prefix is not the same with dir prefix
    suite_list.extend([dir for dir in dir_list \
                            if dir[:2].isdigit()])
    # this is a repetitive check; revise later
    # bug alert: code proceeds if across pairs are defined in input_files
    # for example: if FN and H2 are defined instead of FN,FP or H1,H2
    # might be fixed if checked as tuple pairs
    if trt == 'ASC':
        ALIASES = ['FN', 'Normal', 'H1', 'Hor1', 'FP', 'Parallel', 'H2', 'Hor2']
    else:
        ALIASES = ['SZ1', 'SZ2']
    # calculate RotDnn spectra for each ASC/SZ record set in suite
    suite_rotdnn = manager.dict()
    suite_rotdnn['Periods'] = periods
    inputs = list(
        zip(suite_list, repeat(suite_dir), repeat(ALIASES),
            repeat(periods), repeat(damping_level), repeat(percentile),
            repeat(suite_rotdnn)))
    with mp.Pool() as pool:
        pool.starmap(_record_rotdnn, inputs)

    suite_rotdnn = dict(sorted(suite_rotdnn.items()))
    suite_rotdnn_sorted = {}
    suite_rotdnn_sorted['Periods'] = periods
    suite_rotdnn_sorted = {**suite_rotdnn_sorted, **suite_rotdnn}
    return suite_rotdnn_sorted


def compute_suite_geomean_spectra(suite_dir, periods, damping_level=0.05):
    """
    Calculate Geometric-Mean Acceleration Response Spectra for each record in a
    given suite of acceleration time-histories from SZ (far-field) regions.
    """
    print("Calculating GeoMean-component spectra...")

    dir_list = os.listdir(suite_dir)

    # raise warning for duplicate prefixes?
    for i in np.arange(len(dir_list) + 1):
        suite_list.extend([dir for dir in dir_list \
                            if dir.startswith('{:0>2}'.format(i))])
    # this is a repetitive check; revise later
    SZ_ALIASES = ['SZ']
    # calculate Geometric-Mean spectra for each SZ record set in suite
    suite_gm = {}
    suite_gm['Periods'] = periods
    for record in suite_list:
        record_path = os.path.join(suite_dir, record)
        # this can be removed once repetitive check is fixed
        component_list = os.listdir(record_path)

        matching_alias = [alias in comp for alias in SZ_ALIASES \
                            for comp in component_list]

        if not any(matching_alias):
            # skip GeoMean spectra computation if no SZ file names found in
            # record set dir
            continue

        elif sum(matching_alias) == 1:
            # Raise error message if only 1 matching name is found.
            raise ValueError(   f"No matching record pair found in "
                                f"{os.path.abspath(record)}.")

        # import corresponding time series from record set dir
        record_acc_H1, record_acc_H2, dt = _import_time_series(record_path)
        sax, say = ims.get_response_spectrum_pair(
                        record_acc_H1, dt, record_acc_H2, dt,
                        periods, damping=damping_level, units='g')

        # calculate GeoMean spectra for record pair, but result is in cgs units
        GM_SA_cgs = ims.geometric_mean_spectrum(sax, say)

        # convert to g units
        GM_SA_g = conv(GM_SA_cgs["Pseudo-Acceleration"],
                        from_='cm/s/s', to_='g')

        # append to suite dict
        suite_gm[record] = GM_SA_g

    return suite_gm


# def get_NSCP2015_spectrum():
# Future implementation


# def get_ASCE7-16_spectrum():
# Future implementation


def plot_matching_assessment(save_dir, trt, target, suite,
                                percentile, damping_level=0.05):
    """
    Generates plot of matching assessment for ASC/SZ suite.
    """
    print("Generating " + trt + " matching assessment plot...")

    filename = os.path.join(save_dir, trt + f"_rotd{percentile:.0f}_spectrum.svg")

    if (len(suite) > 1) and target:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        label = trt + " Target Spectra"
        # if percentile == 100.0:
        target['110% SA'] = target['SA'] * 1.10
        label = "110% " + label
        ax.loglog(  target['Periods'],
                    target['110% SA'],
                    'r-',
                    linewidth=3, label=label
                    )
        suite_periods = suite['Periods']
        # Calculate suite average
        Average = gmean(list(suite[k] for k in suite.keys() \
                            if k != 'Periods'))
        ax.loglog(  suite_periods, Average, 'k-', linewidth=3,
                    label=trt + " Suite Average"
                    )

        for record in suite.keys():
            if record != 'Periods':
                ax.loglog(suite_periods, suite[record], '--', linewidth=1,
                            label=record)

        # get maximum value in period list
        ax.axis([min(suite_periods), max(suite_periods), 0.01, 10])
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ticker.ScalarFormatter())
            axis.set_major_formatter(ticker.FormatStrFormatter('%g'))

        ax.set_xlabel("Period T (s)", fontsize=14)
        ax.set_ylabel("Spectral Acceleration $S_a$ (g)", fontsize=14)
        ax.set_title(   trt + f" RotD{percentile:.0f} Response Spectra "
                        f"($\zeta$ = {round(damping_level * 100, 2)}%)",
                        fontsize=18)
        ax.grid(which="both")
        ax.legend(loc=0, fontsize=10)
        fig.savefig(filename, format="svg")

def plot_SZ_geomean(save_dir, SZ_target, SZ_suite,
                                damping_level=0.05):
    """
    Generates plot of matching assessment for SZ suite.
    """
    print("Generating SZ matching assessment plot...")

    filename = os.path.join(save_dir, "SZ_geomean_spectrum.svg")

    # skips plotting if SZ only contains Periods items; I don't like this
    if (len(SZ_suite) > 1) and SZ_target:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.loglog(  SZ_target['Periods'],
                    SZ_target['SA'],
                    'r-',
                    linewidth=3, label="SZ Target Spectra"
                    )

        suite_periods = SZ_suite['Periods']

        # Calculate SZ suite average
        SZ_Average = gmean(list(SZ_suite[k] for k in SZ_suite.keys() \
                            if k != 'Periods'))
        ax.loglog(suite_periods, SZ_Average,
                    'k-', linewidth=3, label="SZ Suite Average"
                    )

        SZ_Average_Max = np.multiply(SZ_Average, 1.1)
        SZ_Average_Min = np.multiply(SZ_Average, 0.9)

        ax.fill_between(suite_periods, SZ_Average_Max, SZ_Average_Min,
                        facecolor='grey', alpha=0.5)

        for record in SZ_suite.keys():
            if record != 'Periods':
                ax.loglog(suite_periods, SZ_suite[record], '--', linewidth=1,
                            label=record)

        ax.axis([min(suite_periods), max(suite_periods), 0.01, 10])
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ticker.ScalarFormatter())
            axis.set_major_formatter(ticker.FormatStrFormatter('%g'))

        ax.set_xlabel("Period T (s)", fontsize=14)
        ax.set_ylabel("Spectral Acceleration $S_a$ (g)", fontsize=14)
        ax.set_title(   f"SZ Geometric-Mean Response Spectra "
                        f"($\zeta$ = {damping_level * 100}%)",
                        fontsize=18)
        ax.grid(which="both")
        ax.legend(loc=0, fontsize=10)
        fig.savefig(filename, format="svg")


def build_save_dir(save_dir):
    """
    Create new directory for placing output file directory.
    """
    if not os.path.exists(save_dir): os.mkdir(save_dir)


def save_data(save_dir, ASC_target, ASC_suite, SZ_target, SZ_suite,
              percentile):
    """
    Save the numerical output data in a single Excel file.
    """
    print(f"Writing output to {os.path.abspath(save_dir)}...")

    # Change directory to output path
    filename = os.path.join(save_dir, "output_matching_assessment.xlsx")

    xlwriter = pd.ExcelWriter(filename)
    df_ASC_suite = pd.DataFrame(ASC_suite)
    df_ASC_target = pd.DataFrame(ASC_target)
    df_SZ_suite = pd.DataFrame(SZ_suite)
    df_SZ_target = pd.DataFrame(SZ_target)

    # Write sheet if not empty
    if not df_ASC_target.empty:
        df_ASC_target.to_excel(xlwriter, sheet_name="ASC Target", index=False)

    # Again, since I was forced to include Periods in dict
    if len(ASC_suite) > 1:
        ASC_Average = gmean(list(ASC_suite[k] for k in ASC_suite.keys() \
                            if k != 'Periods'))
        df_ASC_suite['AVERAGE'] = ASC_Average

        # # Calculate ratios to guide re-matching; still useful?
        # ASC_ratios = np.divide(ASC_target['SA'], ASC_Average)

        df_ASC_suite.to_excel(xlwriter, sheet_name=f"ASC RotD{percentile:.0f}", index=False)
        # df_ASC_suite.to_excel(xlwriter, sheet_name="ASC Ratios", index=False)

    # Write sheet if not empty
    if not df_SZ_target.empty:
        df_SZ_target.to_excel(xlwriter, sheet_name="SZ Target", index=False)

    # Again, since I was forced to include Periods in dict
    if len(SZ_suite) > 1:
        SZ_Average = gmean(list(SZ_suite[k] for k in SZ_suite.keys() \
                            if k != 'Periods'))
        df_SZ_suite['AVERAGE'] = SZ_Average

        # # Calculate ratios to guide re-matching; still useful?
        # SZ_ratios = np.divide(SZ_target['SA'], SZ_Average)

        df_SZ_suite.to_excel(xlwriter, sheet_name=f"SZ RotD{percentile:.0f}", index=False)
        # df_SZ_suite.to_excel(xlwriter, sheet_name="SZ Ratios", index=False)

    xlwriter.close()

def main():
    start_time = time.time()

    ## REQUIRED INPUTS ##
    # # -> for scripting use
    # input_dir = '../data/input_files/NP21.069'
    # # -> default location for dist
    input_dir = os.path.join(os.getcwd(), "data", "input_files")

    damping_ratio = detect_damping_ratio(input_dir)
    percentile = detect_percentile(input_dir)
    periods = np.logspace(-2, 1, num=120)
    ## ####### ####### ##

    # # -> for scripting use
    # output_dir = os.getcwd() + "./output_files"
    # # -> default location for dist
    output_dir = os.path.join(os.getcwd(), "data", "output_files")
    build_save_dir(output_dir)

    # Get target response spectra in suite directory
    ASC_target = import_ASC_target_spectra(input_dir)
    SZ_target = import_SZ_target_spectra(input_dir)

    # Get RotDnn-Component Response Spectra for ASC Suite
    if ASC_target != {}:
        suite_rotdnn_ASC = compute_suite_rotdnn_spectra(input_dir, periods, 
                'ASC', percentile=percentile, damping_level=damping_ratio)
    else: suite_rotdnn_ASC = {}

    # Get RotDnn-Component Response Spectra for SZ Suite
    if SZ_target != {}:
        suite_rotdnn_SZ = compute_suite_rotdnn_spectra(input_dir, periods, 
                'SZ', percentile=percentile, damping_level=damping_ratio)
    else: suite_rotdnn_SZ = {}

    # Get GeoMean-Component Response Spectra for SZ Suite
    # suite_gm_spectra = compute_suite_geomean_spectra(input_dir, periods,
    #                     damping_level=damping_ratio)

    # Plot the output spectra - saved in an output dir
    plot_matching_assessment(output_dir, 'ASC', ASC_target, suite_rotdnn_ASC,
                        percentile=percentile, damping_level=damping_ratio)
    plot_matching_assessment(output_dir, 'SZ', SZ_target, suite_rotdnn_SZ,
                        percentile=percentile, damping_level=damping_ratio)

    # Write the data to spreadsheet
    save_data(output_dir, ASC_target, suite_rotdnn_ASC, SZ_target,
                suite_rotdnn_SZ, percentile)

    end_time = time.time()
    # Log run statistics
    print(  f"Program ran successfully. "
            f"Finished in {(end_time - start_time): .2f} seconds.")

    input("Press ENTER to exit...")


if __name__ == '__main__':
    mp.freeze_support()
    main()
