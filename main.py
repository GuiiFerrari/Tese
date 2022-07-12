import numpy as np
import pandas as pd
import matplotlib as mpl

print(mpl.__version__)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from time import time

# import uproot4 as up
from scipy.signal import find_peaks as fp

# from sklearn.preprocessing import normalize
plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"


def f1():
    num = 310
    name = ""
    data = np.load(f"./nuvem_origi_{num}.npy", allow_pickle=True)
    data_2 = np.load(f"./nuvem_cnn_{num}.npy", allow_pickle=True)
    print(data.shape)
    print(data_2.shape)
    fig = plt.figure(figsize=plt.figaspect(0.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    # ax1 = plt.axes(projection="3d")
    for ax in [ax1, ax2]:
        ax.view_init(27, 111)
        ax.set_zlim((-250, 250))
        ax.set_xlim((-250, 250))
        ax.set_ylim((0, 1000))
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        ax.yaxis.set_major_locator(ticker.MultipleLocator(300))
    ax2.set_title("Traditional method")
    ax1.set_title("CNN reconstructed")
    alphas_1 = data_2[:, 3].reshape(-1)
    alphas_2 = data[:, 3].reshape(-1)
    print(alphas_1.max(), alphas_2.max())
    alphas_1 = alphas_1 / np.sum(alphas_1)
    alphas_2 = alphas_2 / np.sum(alphas_2)
    alphas_1 = alphas_1 / np.max(alphas_1)
    alphas_2 = alphas_2 / np.max(alphas_2)

    ax1.scatter3D(
        data_2[:, 0],
        data_2[:, 2],
        data_2[:, 1],
        marker="s",
        s=4,
        alpha=alphas_1,
    )
    ax2.scatter3D(
        data[:, 0], data[:, 2], data[:, 1], marker="s", s=4, alpha=alphas_2
    )
    plt.tight_layout()
    # print(data)
    # print(data_2)
    # ax1.legend(fontsize = 8)
    # ax2.legend(fontsize = 8)
    plt.show()


def f2():
    df: pd.DataFrame = pd.read_csv(
        "codigo_17F/list_vdrift.txt", sep="\t", header=None
    )
    # df: pd.DataFrame = pd.read_csv(os.path.join(DATA_DIR,
    #     'list_vdrift.txt'), sep='\t', header = None)
    df.columns = ["run", "vdrift", "e1", "e2"]
    del df["e1"]
    del df["e2"]
    df["vdrift"] = df["vdrift"].astype(float) / 3.2
    print(df)
    fig = plt.figure(dpi=200, figsize=(8, 4.5))
    plt.plot(df["run"], df["vdrift"], ls="-", marker="o", markersize=5)
    plt.xlabel("Run")
    # plt.xticks(np.arange(df["run"].min(), df["run"].max(), 10))
    plt.ylabel("Velocidade de deriva (cm/$\mu$s)")
    plt.ylim(0.75, 0.9)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    f2()
