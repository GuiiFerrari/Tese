import tpc_utils as tpc
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.animation as animation

print(mpl.__version__)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from time import time
import uproot4 as up
from scipy.signal import find_peaks as fp
from scipy.fft import fft, ifft, fftfreq, fftshift
import scipy.signal as signal
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 25
plt.rc("axes", titlesize=25, labelsize=25)
plt.rc("xtick", labelsize=25)
plt.rc("ytick", labelsize=25)
xt = np.arange(0.5, 512.0, 1)
import os
from lmfit.models import GaussianModel

plt.rcParams[
    "animation.ffmpeg_path"
] = r"C:\Users\guilh\Documents\ffmpeg-4.4-full_build\ffmpeg-4.4-full_build\bin\ffmpeg.exe"


def load_data_700():
    ti = time()
    # f = up.open("/content/drive/MyDrive/Collabs/tspec_out_192_3.root")
    f = up.open("./dados/tspec_out_224_700.root")
    branches = f[f.keys()[0]].arrays(library="np")
    inputt = []  # branches["input"].reshape(len(branches["input"]), 511)
    target1 = []
    target2 = []  # branches["target"].reshape(len(branches["input"]), 511)
    peaks = []
    BKG = []
    # scaled_deconv = []
    for i in range(len(branches["input"])):
        # inputt.append(branches["input"][i].reshape(-1))
        target1.append(branches["inputBK"][i].reshape(-1))
        target2.append(branches["target"][i].reshape(-1))
        # BKG.append(branches["fundo"][i].reshape(-1))
        # peaks.append(
        #     branches["peaks"][i].astype(float).round().astype(int).reshape(-1)
        # )
    # scaled_deconv.append(branches["deconv_scale"][i].reshape(-1))
    # inputt = np.array(inputt, dtype=float)
    target1 = np.array(target1, dtype=float)
    target2 = np.array(target2, dtype=float)
    # BKG = np.array(BKG, dtype=float)
    # peaks = np.array(peaks, dtype=object)
    # scaled_deconv = np.array(scaled_deconv, dtype = float)
    f.close()
    print("Tempo para carregar dados = %.3fs" % (time() - ti))
    return inputt, target1, target2, peaks, BKG, target2 * 0.85


_, y1, y2, _, _, _ = load_data_700()

fig, ax = plt.subplots(dpi=200)
ax.set_ylim(0, 1000)
ax.set_xlim(0, 512)
ax.set(xlabel="Buckets ($\mu s$)", ylabel="ADC Charge")
ax.grid(True)
num = 69430
x = np.arange(0.5, 512, 1)
data_raw = y1[num]
data = np.array(
    [
        tpc.search_high_res(
            signal=data_raw,
            sigma=4.3,
            threshold=20,
            remove_bkg=False,
            number_it=i,
            markov=False,
            aver_window=3,
        )[0]
        for i in range(701)
    ],
    dtype=float,
)

(line1,) = ax.plot(x, data_raw, lw=1, c="b", alpha=0.5)  # [line1]
(line2,) = ax.plot(x, data_raw, lw=1, c="r")
plt.tight_layout()


def animate(i):
    line1.set_ydata(data_raw)  # update the data.
    line2.set_ydata(data[i])
    return (
        line1,
        line2,
    )


ani = animation.FuncAnimation(fig, animate, frames=700, interval=50)

# # writergif = animation.PillowWriter(fps=60)
FFwriter = animation.FFMpegWriter(fps=60)
ani.save("deconv.mp4", writer=FFwriter)

plt.show()
