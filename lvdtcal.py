import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import os
import pandas as pd
import scipy.linalg as linalg


filename = 'LVDTcalibrationdata_08162023'
data_path = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'random', 'data')
out_path = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'random', 'out')


def get_storage_folder(dirname):
    global dirpath
    print('checking if storage directory exists')
    homefolder = os.path.expanduser('~')
    outfolder = os.path.join('PycharmProjects', 'mcmcrsf_xfiles', 'mcmc_out')
    # name = sim_name

    dirpath = os.path.join(homefolder, outfolder, dirname)
    isExisting = os.path.exists(dirpath)
    if isExisting is False:
        print(f'directory does not exist, creating new directory --> {dirpath}')
        os.makedirs(dirpath)
        return dirpath
    elif isExisting is True:
        print(f'directory exists, all outputs will be saved to existing directory and any existing files will be '
              f'overwritten --> {dirpath}')
        return dirpath


def import_data(datapath, name):
    data = pd.read_csv(f'{os.path.join(datapath, name)}.csv')
    return data


def get_colnames(df):
    cnames = df.columns
    print(cnames)

    return cnames


def plot_data(x, y):
    plt.figure(1)
    plt.plot(x, y, 'k.', label='observed')
    plt.xlabel('displacement (mm)')
    plt.ylabel('current (mA)')
    plt.title('current v. displacement - LVDT calibration')
    plt.legend()
    # plt.show()


def calc_resids(obs, calcd, name):
    r = obs - calcd

    plt.figure(2)
    plt.plot(r, '.', label=f'{name}')
    plt.legend()


def fit_data(x, d, name):
    slope, intercept, r_value, p_value, std_err = st.linregress(x, d)
    d_fit = x * slope + intercept
    plt.figure(1)
    plt.plot(x, d_fit, 'b-', label=f'{name}; {slope:.5f} * x + {intercept:.5f}; r = {r_value:.5f}')
    plt.legend()

    calc_resids(d, d_fit, name)
    print(f'std_err = {std_err}')

    plt.figure(3)
    plt.plot(d, d_fit, '.', label=f'{name}')
    plt.xlabel('observed')
    plt.ylabel('calculated')
    plt.legend()


def main():
    data = import_data(data_path, filename)
    df = pd.DataFrame(data)
    colnames = get_colnames(df)

    x = df['x_mm'].to_numpy()
    cur = df['current_mA'].to_numpy()

    plot_data(x, cur)

    idx_linear = np.where(x >= 7)[0]
    xlinear = x[idx_linear]
    idx_end = np.where(xlinear == np.max(xlinear))[0][0]

    curlinear = cur[idx_linear]

    xlinup = xlinear[0:idx_end]
    xlindown = xlinear[idx_end:]

    curlinup = curlinear[0:idx_end]
    curlindown = curlinear[idx_end:]

    fit_data(xlinear, curlinear, 'all')
    fit_data(xlinup, curlinup, 'up')
    fit_data(xlindown, curlindown, 'down')

    plt.show()


if __name__ == '__main__':
    main()