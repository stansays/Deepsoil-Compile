import intensity_measures as ims # Intensity Measure Tools
import pandas as pd
import numpy as np
import os

def parse_record_txt(file):
    with open(file) as f:
        lines = f.readlines()
        tab = lines[2].find('\t')
        time_step = float(lines[2][:tab])
        record = np.array(pd.read_csv(file, delimiter='\t').iloc[:, 1])
        return record, time_step


periods = np.array([
    0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15,
    0.16, 0.17, 0.18, 0.19, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34,
    0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
    0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
    2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8,
    5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0
],
                   dtype=float)

cwd = os.path.abspath('../data/input/ATH/')
text_files = [f for f in os.listdir(cwd) if f.endswith('.txt')]
for i in range(0, len(text_files), 2):
    df = pd.DataFrame()
    x_record, x_time_step = parse_record_txt('../data/input/ATH/' + text_files[i])
    y_record, y_time_step = parse_record_txt('../data/input/ATH/' + text_files[i+1])
    gmrotd100 = ims.gmrotdpp(x_record, x_time_step, y_record, y_time_step,
                        periods, percentile=100.0, damping=0.05)
    df['Period'] = gmrotd100['periods']
    df['Sa'] = gmrotd100['GMRotDpp']
    df.to_csv('../data/output/ATH/' + text_files[i][:2] + '.csv', index=False)
