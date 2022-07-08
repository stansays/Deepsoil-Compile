'''
Python script to read the recorded timehistory from NGA database
Modified for Python 3.X

input : path to the timehistory file
output : the acceleration time history with dt

(c) Shrey Shahi
https://github.com/shreyshahi/parseAT2
'''
import os
import numpy as np

def parse(fname):
    if not os.path.exists(fname):
        return {'Acc' : -1, 'dt':-1 , 'NPTS':-1, 'error':-1}

    accFile = open(fname)
    # Burn through the first 3 lines
    for i in range(3):
        burn = accFile.readline()

    infoLine = accFile.readline()
    data = infoLine.split(',')
    NPTS = int(data[0].split('=')[1].strip())
    dt = float(data[1].split('=')[1].strip().split(' ')[0].strip())
    acc = []
    for line in accFile:
        data = line.split()
        data = [float(d.strip()) for d in data if len(d) > 0]
        acc += data

    assert len(acc) == NPTS

    return {'Acc':acc , 'dt': dt, 'NPTS':NPTS, 'error':0}

def main():
    input_dir = "C:\\Users\\francis.bernales\OneDrive - AMH Philippines, Inc\\NP22.063 Chodai 4th Cebu-Mactan Bridge SHA\\06 NP22.063 WORK FILES\\06 SRA (L2)\\01 Input Motions"
    output_dir = "C:\\Users\\francis.bernales\OneDrive - AMH Philippines, Inc\\NP22.063 Chodai 4th Cebu-Mactan Bridge SHA\\06 NP22.063 WORK FILES\\06 SRA (L2)\\02 DEEPSOIL\\Input Motions"
    os.mkdir(os.path.join(output_dir, output_dir))
      
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".AT2"):
                parsed_dict = parse(os.path.join(root, file))
                ACC = np.array(parsed_dict['Acc'])
                NPTS = parsed_dict['NPTS']
                DT = parsed_dict['dt']

                t = np.linspace(DT,DT*NPTS,NPTS)
                
                with open(os.path.join(output_dir, os.path.splitext(file)[0] + ".txt"), 'a') as f:
                    f.write(str(NPTS) + "  " + str(DT) + "\n")
                    np.savetxt(f, np.c_[t,ACC], fmt='%.6e')




if __name__ == '__main__':
    main()