#!/usr/bin/env python3

# AMH Philippines Inc.
# FJTBernales (2021.07.30)

'''
Combines scaled or spectrally-matched horizontal-component time histories and
compares with a target spectra.
'''

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.stats.mstats import gmean
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

    H1_record = pd.read_csv(os.path.join(folder_acc_data, H1_file), sep='\t')
    H2_record = pd.read_csv(os.path.join(folder_acc_data, H2_file), sep='\t')

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
                            is not consistent!")

    if not np.all(np.fabs(np.diff(H2_time) - time_step) < 1E-10):
        raise ValueError(f"Time step in {os.path.abspath(H2_file)} \
                            is not consistent!")

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

    # inefficient search and case-sensitive;
    # improve with regex; check for multiple matches
    targets = [comp for comp in component_list if 'target' in comp.lower()]
    if not targets:
        raise IndexError("Provide target spectra files!")

    # Search for ASC target
    # bug alert: code proceeds if only 1 target file but no ASC nor SZ specified
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
        df = pd.read_csv(os.path.join(suite_dir, targ_file), sep='\t')

        # check for missing fields and raise error

        ASC_target = {}
        ASC_target['Periods'] = df.iloc[:, 0].to_numpy()
        ASC_target['SA'] = df.iloc[:, 1].to_numpy()

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

    # inefficient search and case-sensitive;
    # improve with regex; check for multiple matches
    targets = [comp for comp in component_list if 'target' in comp.lower()]
    if not targets:
        raise IndexError("Provide target spectra files!")

    # Search for SZ target
    # bug alert: code proceeds if only 1 target file but no ASC nor SZ specified
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
        df = pd.read_csv(os.path.join(suite_dir, targ_file), sep='\t')

        # check for missing fields and raise error

        SZ_target = {}
        SZ_target['Periods'] = df.iloc[:, 0].to_numpy()
        SZ_target['SA'] = df.iloc[:, 1].to_numpy()

    return SZ_target


def compute_suite_rotd100_spectra(suite_dir, periods, damping_level=0.05):
    """
    Calculate RotD100 Response Spectra for each record in a given suite of
    acceleration time-histories from ASC regions.
    """
    dir_list = os.listdir(suite_dir)

    # raise warning for duplicate prefixes?
    # raise warning if file prefix is not the same with dir prefix
    for i in np.arange(len(dir_list) + 1):
        suite_list.extend([dir for dir in dir_list \
                            if dir.startswith('{:0>2}'.format(i))])
    # this is a repetitive check; revise later
    # bug alert: code proceeds if across pairs are defined in input_files
    # for example: if FN and H2 are defined instead of FN,FP or H1,H2
    # might be fixed if checked as tuple pairs
    ASC_ALIASES = ['FN', 'Normal', 'H1', 'Hor1', 'FP', 'Parallel', 'H2', 'Hor2']
    # calculate RotD100 spectra for each ASC record set in suite
    suite_rotd100 = {}
    suite_rotd100['Periods'] = periods
    for record in suite_list:
        record_path = os.path.join(suite_dir, record)
        # this can be removed once repetitive check is fixed
        component_list = os.listdir(record_path)

        matching_alias = [alias in comp for alias in ASC_ALIASES \
                            for comp in component_list]

        if not any(matching_alias):
            # skip RotD100 computation if no ASC file names found in record dir
            continue

        elif sum(matching_alias) == 1:
            # Raise error message if only 1 matching name is found.
            raise ValueError(   f"No matching record pair found in "
                                f"{os.path.abspath(record)}.")
        # import corresponding time series from record set dir
        record_acc_H1, record_acc_H2, dt = _import_time_series(record_path)

        # calculate RotD100 output dict for record, but result is in cgs units
        RotD100_cgs = ims.rotdpp(record_acc_H1, dt, record_acc_H2, dt,
                                periods, percentile=100.0,
                                damping=damping_level, units='g')[0]
        # # FOR WORKFLOW TESTING ONLY
        # sax, say = ims.get_response_spectrum_pair(
        #                 record_acc_H1, dt, record_acc_H2, dt,
        #                 periods, damping=damping_level, units='g')
        #
        # # calculate GeoMean spectra for record pair, but result is in cgs units
        # RotD100_cgs = ims.geometric_mean_spectrum(sax, say)

        # convert to g units
        RotD100_SA_g = conv(RotD100_cgs["Pseudo-Acceleration"],
                            from_='cm/s/s', to_='g')
        # append to suite dict
        suite_rotd100[record] = RotD100_SA_g

    return suite_rotd100


def compute_suite_geomean_spectra(suite_dir, periods, damping_level=0.05):
    """
    Calculate Geometric-Mean Acceleration Response Spectra for each record in a
    given suite of acceleration time-histories from SZ (far-field) regions.
    """
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


def plot_ASC_matching_assessment(save_dir, ASC_target, ASC_suite,
                                damping_level=0.05):
    """
    Generates plot of matching assessment for ASC suite.
    """
    filename = os.path.join(save_dir, "ASC_rotd100_spectrum.svg")

    if (len(ASC_suite) > 1) and ASC_target:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.loglog(  ASC_target['Periods'],
                    ASC_target['SA'],
                    'r-',
                    linewidth=3, label="ASC Target Spectra"
                    )
        suite_periods = ASC_suite['Periods']
        # Calculate ASC suite average
        ASC_Average = gmean(list(ASC_suite[k] for k in ASC_suite.keys() \
                            if k != 'Periods'))
        ax.loglog(  suite_periods, ASC_Average, 'k-', linewidth=3,
                    label="ASC Suite Average"
                    )

        for record in ASC_suite.keys():
            if record != 'Periods':
                ax.loglog(suite_periods, ASC_suite[record], '--', linewidth=1,
                            label=record)

        # get maximum value in period list
        ax.axis([min(suite_periods), max(suite_periods), 0.01, 10])
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ticker.ScalarFormatter())
            axis.set_major_formatter(ticker.FormatStrFormatter('%g'))

        ax.set_xlabel("Period T (s)", fontsize=14)
        ax.set_ylabel("Spectral Acceleration $S_a$ (g)", fontsize=14)
        ax.set_title(   f"ASC Maximum-Direction Response Spectra "
                        f"($\zeta$ = {damping_level * 100}%)",
                        fontsize=18)
        ax.grid(which="both")
        ax.legend(loc=0, fontsize=10)
        fig.savefig(filename, format="svg")


def plot_SZ_matching_assessment(save_dir, SZ_target, SZ_suite,
                                damping_level=0.05):
    """
    Generates plot of matching assessment for SZ suite.
    """
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


def set_cwd_to_src_loc():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


def save_data(save_dir, ASC_target, ASC_suite, SZ_target, SZ_suite):
    """
    Output the numerical output data in a single Excel file.
    """
    # Change directory to output path
    filename = os.path.join(save_dir, "output_matching_assessment.xlsx")

    xlwriter = pd.ExcelWriter(filename)
    df_ASC_suite = pd.DataFrame(ASC_suite)
    df_ASC_target = pd.DataFrame(ASC_target)
    df_SZ_suite = pd.DataFrame(SZ_suite)
    df_SZ_target = pd.DataFrame(SZ_target)

    # Write sheet if not empty
    if not df_ASC_target.empty:
        df_ASC_target["110% SA"] = 1.1 * df_ASC_target["SA"]
        df_ASC_target.to_excel(xlwriter, sheet_name="ASC Target", index=False)

    # Again, since I was forced to include Periods in dict
    if len(ASC_suite) > 1:
        ASC_Average = gmean(list(ASC_suite[k] for k in ASC_suite.keys() \
                            if k != 'Periods'))
        df_ASC_suite['AVERAGE'] = ASC_Average

        # # Calculate ratios to guide re-matching; still useful?
        # ASC_ratios = np.divide(ASC_target['SA'], ASC_Average)

        df_ASC_suite.to_excel(xlwriter, sheet_name="ASC RotD100", index=False)
        # df_ASC_suite.to_excel(xlwriter, sheet_name="ASC Ratios", index=False)

    # Write sheet if not empty
    if not df_SZ_target.empty:
        df_SZ_target["110% SA"] = 1.1 * df_SZ_target["SA"]
        df_SZ_target.to_excel(xlwriter, sheet_name="SZ Target", index=False)

    # Again, since I was forced to include Periods in dict
    if len(SZ_suite) > 1:
        SZ_Average = gmean(list(SZ_suite[k] for k in SZ_suite.keys() \
                            if k != 'Periods'))
        df_SZ_suite['AVERAGE'] = SZ_Average

        # # Calculate ratios to guide re-matching; still useful?
        # SZ_ratios = np.divide(SZ_target['SA'], SZ_Average)

        df_SZ_suite.to_excel(xlwriter, sheet_name="SZ GeoMean", index=False)
        # df_SZ_suite.to_excel(xlwriter, sheet_name="SZ Ratios", index=False)

    xlwriter.close()

def main():
    ## REQUIRED INPUTS ##
    input_dir = '../'
    damping_ratio = 0.05
    periods = np.array([0.01, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 2, 3, 4,
                        5, 7.5, 10], dtype=float)
    ## ####### ####### ##

    # Change current working directory ("cwd") to location of script or dist
    set_cwd_to_src_loc()
    # output_dir = os.path.join(wd, "data/output_files") -> for scripting use
    output_dir = os.getcwd() + "/output_files" # -> default location for dist
    build_save_dir(output_dir)

    # Get target response spectra in suite directory
    ASC_target = import_ASC_target_spectra(input_dir)
    SZ_target = import_SZ_target_spectra(input_dir)

    # Get RotD1000-Component Response Spectra for ASC Suite
    suite_rotd100_spectra = compute_suite_rotd100_spectra(input_dir, periods,
                                                    damping_level=damping_ratio)

    # Get GeoMean-Component Response Spectra for SZ Suite
    suite_gm_spectra = compute_suite_geomean_spectra(input_dir, periods,
                                                    damping_level=damping_ratio)

    # Plot the output spectra - saved in an output dir
    plot_ASC_matching_assessment(output_dir, ASC_target, suite_rotd100_spectra,
                                damping_level=damping_ratio)
    plot_SZ_matching_assessment(output_dir, SZ_target, suite_gm_spectra,
                                damping_level=damping_ratio)

    # Write the data to spreadsheet
    save_data(output_dir, ASC_target, suite_rotd100_spectra, SZ_target,
                suite_gm_spectra)

if __name__ == '__main__':
    main()