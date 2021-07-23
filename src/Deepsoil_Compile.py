import sqlite3
import os
import math
import pandas as pd
import numpy as np
import multiprocessing as mp


def merge_profile(profile):
    cwd = os.path.abspath('../' + profile)
    folders = [f for f in os.listdir(cwd) if f.startswith('Motion_')]

    df_surface = pd.DataFrame()
    df_input = pd.DataFrame()
    df_disp = pd.DataFrame()
    df_strain = pd.DataFrame()
    df_stress = pd.DataFrame()

    for folder in folders:
        pos = folder.find('(') - 1
        motion = folder[7:pos]
        conn = sqlite3.connect('../' + profile + '/' + folder +
                               '/deepsoilout.db3')

        # Input Motion
        df_next_input = pd.read_sql_query(
            'SELECT PERIOD, INPUT_MOTION_RS FROM RESPONSE_SPECTRA', conn)
        df_next_input = df_next_input.rename(columns={
            'INPUT_MOTION_RS': motion
        }).set_index('PERIOD')
        df_input = df_next_input.join(df_input)

        # Surface Motion
        df_next_surf = pd.read_sql_query(
            'SELECT PERIOD, LAYER1_RS FROM RESPONSE_SPECTRA', conn)
        df_next_surf = df_next_surf.rename(columns={
            'LAYER1_RS': motion
        }).set_index('PERIOD')
        df_surface = df_next_surf.join(df_surface)

        # Displacement
        df_next_disp = pd.read_sql_query(
            'SELECT DEPTH_LAYER_TOP, MIN_DISP_RELATIVE, MAX_DISP_RELATIVE FROM PROFILES',
            conn)
        df_next_disp = df_next_disp.abs()
        df_next_disp[motion] = df_next_disp[[
            'MIN_DISP_RELATIVE', 'MAX_DISP_RELATIVE'
        ]].max(axis=1)

        df_next_disp = df_next_disp.drop(
            columns=['MIN_DISP_RELATIVE', 'MAX_DISP_RELATIVE']).set_index(
                'DEPTH_LAYER_TOP')
        df_disp = df_next_disp.join(df_disp)

        # Strain (%)
        df_next_strain = pd.read_sql_query(
            'SELECT DEPTH_LAYER_MID, MAX_STRAIN FROM PROFILES', conn)
        df_next_strain = df_next_strain.rename(columns={
            'MAX_STRAIN': motion
        }).set_index('DEPTH_LAYER_MID')
        df_strain = df_next_strain.join(df_strain)

        # Stress Ratio
        df_next_stress = pd.read_sql_query(
            'SELECT DEPTH_LAYER_MID, MAX_STRESS_RATIO FROM PROFILES', conn)
        df_next_stress = df_next_stress.rename(columns={
            'MAX_STRESS_RATIO': motion
        }).set_index('DEPTH_LAYER_MID')
        df_stress = df_next_stress.join(df_stress)

    df_input = df_input.reindex(sorted(df_input.columns), axis=1)
    df_input_mean = df_input.groupby(df_input.columns.str[:2],
                                     axis=1).prod().pow(0.5)

    n_suite = len(df_input_mean.columns)

    # Input Mean
    df_input['Mean'] = df_input.prod(axis=1).pow(0.5 / n_suite)
    df_input_mean['Mean'] = df_input_mean.prod(axis=1).pow(1. / n_suite)

    # Surface Mean
    df_surface = df_surface.reindex(sorted(df_surface.columns), axis=1)
    df_surface_mean = df_surface.groupby(df_surface.columns.str[:2],
                                         axis=1).prod().pow(0.5)
    df_surface['Mean'] = df_surface.prod(axis=1).pow(0.5 / n_suite)
    df_surface_mean['Mean'] = df_surface_mean.prod(axis=1).pow(1. / n_suite)

    # Amplification Mean
    df_ampl = df_surface_mean.iloc[:, :-1] / df_input_mean.iloc[:, :-1]
    df_ampl_xim = df_surface_mean.iloc[:, :-1] / df_input_mean.iloc[0][:-1]
    df_ampl['Mean'] = df_ampl.prod(axis=1).pow(1. / n_suite)
    df_ampl_xim['Mean'] = df_ampl_xim.prod(axis=1).pow(1. / n_suite)

    # Displacement Mean
    df_disp = df_disp.reindex(sorted(df_disp.columns), axis=1)
    df_disp['Mean'] = df_disp.mean(axis=1)

    # Strain Mean
    df_strain = df_strain.reindex(sorted(df_strain.columns), axis=1)
    df_strain['Mean'] = df_strain.mean(axis=1)

    # Stress Ratio Mean
    df_stress = df_stress.reindex(sorted(df_stress.columns), axis=1)
    df_stress['Mean'] = df_stress.mean(axis=1)

    # Write RS to Excel
    writer_SRA = pd.ExcelWriter('../' + profile + '/' + profile + '_RS.xlsx')
    df_input.to_excel(writer_SRA, 'Input Motion')
    df_input_mean.to_excel(writer_SRA, 'Input GM Spectra')
    df_surface.to_excel(writer_SRA, 'Surface Motion')
    df_surface_mean.to_excel(writer_SRA, 'Surface GM Spectra')
    df_ampl.to_excel(writer_SRA, 'Amplification Spectra')
    df_ampl_xim.to_excel(writer_SRA, 'Amplification x_IM,ref')
    writer_SRA.save()

    # Write Profile to Excel
    writer_Profile = pd.ExcelWriter('../' + profile + '/' + profile +
                                    '_Profile.xlsx')
    df_disp.to_excel(writer_Profile, 'Displacement')
    df_strain.to_excel(writer_Profile, 'Strain')
    df_stress.to_excel(writer_Profile, 'Stress Ratio')
    writer_Profile.save()


def df_next_comb(xlsx, df, sheet_name, mean_col):
    df_next = pd.read_excel(xlsx,
                            sheet_name=sheet_name).rename(columns={
                                'Mean': mean_col,
                            })
    df = df_next[['PERIOD', mean_col]].set_index('PERIOD').join(df)
    return df


def main():
    cwd = os.path.abspath('..')
    profiles = os.listdir(cwd)
    profiles = [f for f in profiles if f.startswith('profile_')]
    with mp.Pool() as pool:
        pool.map(merge_profile, profiles)

    df_surf_comb = pd.DataFrame()
    df_ampl_comb = pd.DataFrame()
    df_ampl_xim_comb = pd.DataFrame()
    df_disp_comb = pd.DataFrame()
    df_strain_comb = pd.DataFrame()
    df_stress_comb = pd.DataFrame()
    df_surf_GM = pd.DataFrame()
    df_ampl_GM = pd.DataFrame()
    df_ampl_xim_GM = pd.DataFrame()
    pd.options.mode.use_inf_as_na = True

    for profile in reversed(profiles):
        mean_col = 'Mean ' + profile[8:]

        # Compile RS
        xlsx_RS = pd.ExcelFile('../' + profile + '/' + profile + '_RS.xlsx')
        df_surf_comb = df_next_comb(xlsx_RS, df_surf_comb,
                                'Surface GM Spectra', mean_col)
        df_ampl_comb = df_next_comb(xlsx_RS, df_ampl_comb,
                                'Amplification Spectra', mean_col)
        df_ampl_xim_comb = df_next_comb(xlsx_RS, df_ampl_xim_comb,
                                'Amplification x_IM,ref', mean_col)
        if df_surf_GM.empty:
            df_surf_GM = pd.read_excel(xlsx_RS,
                            sheet_name='Surface GM Spectra', index_col=0)
            df_ampl_GM = pd.read_excel(xlsx_RS,
                            sheet_name='Amplification Spectra', index_col=0)
            df_ampl_xim_GM = pd.read_excel(xlsx_RS,
                            sheet_name='Amplification x_IM,ref', index_col=0)
        else:
            df_surf_GM = df_surf_GM.mul(pd.read_excel(
                xlsx_RS, sheet_name='Surface GM Spectra', index_col=0),
                                        fill_value=1)
            df_ampl_GM = df_ampl_GM.mul(pd.read_excel(
                xlsx_RS, sheet_name='Amplification Spectra', index_col=0),
                                        fill_value=1)
            df_ampl_xim_GM = df_ampl_xim_GM.mul(pd.read_excel(
                xlsx_RS, sheet_name='Amplification x_IM,ref', index_col=0),
                                        fill_value=1)

        # Compile Profile
        xlsx_PF = pd.ExcelFile('../' + profile + '/' + profile +
                               '_Profile.xlsx')

        # Compile Strain Tab
        df_next_strain_comb = pd.read_excel(xlsx_PF, sheet_name='Strain')
        df_next_strain_comb = df_next_strain_comb[[
            'DEPTH_LAYER_MID', 'Mean'
        ]].rename(columns={
            'DEPTH_LAYER_MID': 'Depth',
            'Mean': mean_col
        })
        max_depth = math.ceil(df_next_strain_comb['Depth'].iloc[-1])
        df_depths = pd.DataFrame(np.arange(0.5, max_depth, 1),
                                 columns=['Depth'])
        df_next_strain_comb = pd.merge_asof(df_depths,
                                            df_next_strain_comb,
                                            on='Depth')
        df_strain_comb = df_next_strain_comb.set_index('Depth').join(
            df_strain_comb)

        # Compile Stress Ratio Tab
        df_next_stress_comb = pd.read_excel(xlsx_PF, sheet_name='Stress Ratio')
        df_next_stress_comb = df_next_stress_comb[[
            'DEPTH_LAYER_MID', 'Mean'
        ]].rename(columns={
            'DEPTH_LAYER_MID': 'Depth',
            'Mean': mean_col
        })
        df_next_stress_comb = pd.merge_asof(df_depths,
                                            df_next_stress_comb,
                                            on='Depth')
        df_stress_comb = df_next_stress_comb.set_index('Depth').join(
            df_stress_comb)

        # Compile Displacement Tab
        df_next_disp_comb = pd.read_excel(xlsx_PF, sheet_name='Displacement')
        df_next_disp_comb = df_next_disp_comb[['DEPTH_LAYER_TOP', 'Mean'
                                               ]].rename(columns={
                                                   'DEPTH_LAYER_TOP': 'Depth',
                                                   'Mean': mean_col
                                               })
        df_next_disp_comb = pd.merge_asof(df_depths,
                                          df_next_disp_comb,
                                          on='Depth')
        df_disp_comb = df_next_disp_comb.set_index('Depth').join(df_disp_comb)

    # Sort Profiles
    profile_order = [int(profile[8:]) for profile in profiles]
    zipped = zip(profile_order, df_surf_comb.columns)
    sorted_zipped_lists = sorted(zipped)
    sorted_cols = [element for _, element in sorted_zipped_lists]

    df_surf_comb = df_surf_comb.reindex(sorted_cols, axis=1)
    df_ampl_comb = df_ampl_comb.reindex(sorted_cols, axis=1)
    df_ampl_xim_comb = df_ampl_xim_comb.reindex(sorted_cols, axis=1)
    df_disp_comb = df_disp_comb.reindex(sorted_cols, axis=1)
    df_strain_comb = df_strain_comb.reindex(sorted_cols, axis=1)
    df_stress_comb = df_stress_comb.reindex(sorted_cols, axis=1)

    # RS Geomean for batch run
    n_valid_profiles = df_surf_comb.count(axis=1)
    df_surf_comb['Sa (g)'] = df_surf_comb.prod(axis=1).pow(
        1. / n_valid_profiles)
    df_ampl_comb['Sa (g)'] = df_ampl_comb.prod(axis=1).pow(
        1. / n_valid_profiles)
    df_ampl_xim_comb['Sa (g)'] = df_ampl_xim_comb.prod(axis=1).pow(
        1. / n_valid_profiles)
    df_surf_GM = df_surf_GM.pow(1. / n_valid_profiles.mean())
    df_ampl_GM = df_ampl_GM.pow(1. / n_valid_profiles.mean())
    df_ampl_xim_GM = df_ampl_xim_GM.pow(1. / n_valid_profiles.mean())

    # Write RS_Merged
    writer_Merged_RS = pd.ExcelWriter('../RS_Merged.xlsx')
    df_surf_comb.to_excel(writer_Merged_RS, 'Surface GM Spectra')
    df_ampl_comb.to_excel(writer_Merged_RS, 'Amplification Spectra')
    df_ampl_xim_comb.to_excel(writer_Merged_RS, 'Amplification x_IM,ref')
    writer_Merged_RS.save()

    # Write GMs_Merged
    writer_Merged_GMs = pd.ExcelWriter('../GMs_Merged.xlsx')
    df_surf_GM.to_excel(writer_Merged_GMs, 'Surface GM Spectra')
    df_ampl_GM.to_excel(writer_Merged_GMs, 'Amplification Spectra')
    df_ampl_xim_GM.to_excel(writer_Merged_GMs, 'Amplification x_IM,ref')
    writer_Merged_GMs.save()

    # Write Profile_Merged
    writer_Merged_Prof = pd.ExcelWriter('../Profile_Merged.xlsx')
    df_disp_comb.to_excel(writer_Merged_Prof, 'Displacement')
    df_strain_comb.to_excel(writer_Merged_Prof, 'Strain')
    df_stress_comb.to_excel(writer_Merged_Prof, 'Stress Ratio')
    writer_Merged_Prof.save()


if __name__ == '__main__':
    main()
