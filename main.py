from turtle import color
import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.interpolate import CubicSpline as CS
import os

# print(mpl.__version__)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from time import time

import uproot4 as up
from scipy.signal import find_peaks as fp

# from sklearn.preprocessing import normalize
plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
np.seterr(all="ignore")


def inc_dist(w, contagens, inc_lum=0):
    """
    Calcula a incerteza das distribuições angulares.
    """
    lum = 1
    return w * (np.sqrt(contagens) / contagens)
    # return w * 0.23
    # return w * np.sqrt(
    #     (np.sqrt(contagens) / contagens) ** 2
    #     + (0.23) ** 2
    #     # + (inc_lum / lum) ** 2
    # )


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


def f3():
    #          0, 1,  2,  3,  4,  5,  6,  7,  8,  9
    numbers = np.arange(40, 311, 10)
    keys1 = [f"Th_zprof/Theta_hist_{d};1" for d in numbers]
    keys2 = [f"Th_zprof/Theta_hist_C_{d};1" for d in numbers]
    keys3 = [f"Th_zprof/EBeam_hist_{d};1" for d in numbers]
    keys4 = [f"Th_zprof/contagens_{d};1" for d in numbers]
    keys5 = [f"Th_zprof/contagens_C_{d};1" for d in numbers]
    # for index in [5, 7, 12, 16]:
    for index in range(len(numbers)):
        with up.open("somas/soma_CS.root") as file:
            # index = np.random.randint(low=0, high=len(keys1), size=1)[0]
            # index = 16
            # print(index)
            w1, xbins1 = file[keys1[index]].to_numpy()  # Inclusivo
            # print(w1, xbins1)
            w2, xbins2 = file[keys2[index]].to_numpy()  # Exclusivo
            # print(w2, xbins2)
            w3, xbins3 = file[keys3[index]].to_numpy()  # Energia
            # print(w3, xbins3)
            w4, xbins4 = file[keys4[index]].to_numpy()  # Contagens inclusivo
            # print(w4, xbins4)
            w5, xbins5 = file[keys5[index]].to_numpy()  # Contagens exclusivo
            # print(w5, xbins5)
        # Energia do beam
        centers = (xbins3[1:] + xbins3[:-1]) / 2
        # print(len(centers), len(w3))
        # média ponderada
        Ebeam = np.sum(centers * w3) / np.sum(w3)
        Ebeam = Ebeam * (4 / (4 + 17))
        print(Ebeam)
        fig = plt.figure(dpi=180, figsize=(8, 4.5))
        # centers1 = (xbins1[0:] + xbins1[:-1]) / 2
        centers1 = np.zeros(len(xbins1) - 1)
        centers2 = np.zeros(len(xbins2) - 1)
        # print(xbins1)
        for i in range(len(centers1)):
            centers1[i] = (xbins1[i] + xbins1[i + 1]) / 2
            centers2[i] = (xbins2[i] + xbins2[i + 1]) / 2

        # centers2 = (xbins2[0:] + xbins2[:-1]) / 2
        # a = np.nan_to_num(np.sqrt(w4) / w4)
        # b = np.nan_to_num(np.sqrt(w5) / w5)
        # a = np.sqrt(w4) / w4
        # b = np.sqrt(w5) / w5
        # a = a[~np.isnan(a)]
        # b = b[~np.isnan(b)]
        # print(a)
        # print(b)
        # print(
        #     f"Razão do erro inclusivo: média = {a.mean():.4f}, min = {a.min():.4f}, max = {a.max():.4f}"
        # )
        # print(
        #     f"Razão do erro exclusivo: média = {b.mean():.4f}, min = {b.min():.4f}, max = {b.max():.4f}"
        # )
        w1 = 0.2475 * w1
        w1[:5] = 0
        w1[np.where(w1 == 0)[0]] = np.nan
        inc_w1 = inc_dist(w1, w4)
        d1 = {"angulo": centers1, "xs": w1, "contagens": w4, "inc_w1": inc_w1}
        df1 = pd.DataFrame(d1)
        # pd.DataFrame.to_csv(df1, f"csvs/inclusivo_E_{Ebeam}_.csv")
        # print(df1)
        w2 = 0.2475 * w2
        w2[:5] = 0
        w2[np.where(w2 == 0)[0]] = np.nan
        inc_w2 = inc_dist(w2, w5)
        d2 = {"angulo": centers2, "xs": w2, "contagens": w5, "inc_w2": inc_w2}
        df2 = pd.DataFrame(d2)
        # pd.DataFrame.to_csv(df2, f"csvs/exclusivo_E_{Ebeam}_.csv")
        plt.errorbar(
            centers1,
            w1,
            ms=5,
            yerr=inc_w1,
            ecolor="black",
            fmt="",
            capsize=5,
            ls="",
            marker="o",
            c="black",
            label="Inclusivo",
        )
        plt.errorbar(
            centers2,
            w2,
            ms=5,
            yerr=inc_w2,
            ecolor="red",
            fmt="",
            capsize=5,
            ls="",
            marker="o",
            c="red",
            label="Exclusivo",
        )
        # cs_in = CS(centers1, w1)
        # cs_ex = CS(centers2, w2)
        # xs = np.linspace(0, 90, 1000)
        # plt.plot(xs, cs_in(xs), c="black", ls="--")
        # plt.plot(xs, cs_ex(xs), c="red", ls="--")
        plt.xlabel(r"$\theta_p$ (graus)")
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))
        plt.xlim(0, 90)
        plt.ylabel("$d\sigma$/d$\Omega$ (mb/sr)")
        plt.ylim(bottom=0.0)
        if index == 0:
            plt.legend(framealpha=1.0, edgecolor="black")
            plt.annotate(
                "$E_{CM}$ = " + f"{Ebeam:.2f} MeV",
                xy=(0.62, 0.55),
                xycoords="axes fraction",
                color="black",
            )
        else:
            plt.annotate(
                "$E_{CM}$ = " + f"{Ebeam:.2f} MeV",
                xy=(0.62, 0.775),
                xycoords="axes fraction",
                color="black",
            )
        # plt.tight_layout(rect=(0.115, 0.163, 0.980, 0.988))
        plt.tight_layout()
        plt.savefig(
            f"figs/dist_angs/dist_ang_{index}", dpi=600, bbox_inches="tight"
        )
        plt.close()
        # plt.show()


def f5():
    numbers = np.arange(40, 311, 10)
    keys1 = [f"Th_zprof/Theta_hist_{d};1" for d in numbers]
    keys2 = [f"Th_zprof/Theta_hist_C_{d};1" for d in numbers]
    keys3 = [f"Th_zprof/EBeam_hist_{d};1" for d in numbers]
    keys4 = [f"Th_zprof/contagens_{d};1" for d in numbers]
    keys5 = [f"Th_zprof/contagens_C_{d};1" for d in numbers]
    # for index in [5, 7, 12, 16]:
    energias = []
    contagens_I = []
    contagens_E = []
    incertezas_I = []
    incertezas_E = []
    for index in range(len(numbers)):
        with up.open("somas/soma_CS.root") as file:
            # index = np.random.randint(low=0, high=len(keys1), size=1)[0]
            # index = 16
            # print(index)
            w1, xbins1 = file[keys1[index]].to_numpy()  # Inclusivo
            # print(w1, xbins1)
            w2, xbins2 = file[keys2[index]].to_numpy()  # Exclusivo
            # print(w2, xbins2)
            w3, xbins3 = file[keys3[index]].to_numpy()  # Energia
            # print(w3, xbins3)
            w4, xbins4 = file[keys4[index]].to_numpy()  # Contagens inclusivo
            # print(w4, xbins4)
            w5, xbins5 = file[keys5[index]].to_numpy()  # Contagens exclusivo
            # print(w5, xbins5)
        # Energia do beam
        centers = (xbins3[1:] + xbins3[:-1]) / 2
        # print(len(centers), len(w3))
        # média ponderada
        Ebeam = np.sum(centers * w3) / np.sum(w3)
        # energias.append(Ebeam)
        energias.append(Ebeam * (4 / (4 + 17)))
        contagens_I.append(np.sum(w4))
        contagens_E.append(np.sum(w5))
        incertezas_I.append(np.sqrt(np.sum(np.sqrt(w4) ** 2)))
        incertezas_E.append(np.sqrt(np.sum(np.sqrt(w5) ** 2)))
    contagens_I = np.array(contagens_I) * 0.2475
    contagens_E = np.array(contagens_E) * 0.2475
    incertezas_I = np.array(incertezas_I)
    incertezas_E = np.array(incertezas_E)
    fig = plt.figure(dpi=180, figsize=(8, 4.5))
    # plt.scatter(energias, contagens_I, c="black", label="Inclusivo")
    # plt.scatter(energias, contagens_E, c="red", label="Exclusivo")
    plt.errorbar(
        energias,
        contagens_I,
        ms=5,
        yerr=incertezas_I,
        ecolor="black",
        fmt="",
        capsize=5,
        ls="",
        marker="o",
        c="black",
        label="Inclusivo",
    )
    plt.errorbar(
        energias,
        contagens_E,
        ms=5,
        yerr=incertezas_E,
        ecolor="red",
        fmt="",
        capsize=5,
        ls="",
        marker="o",
        c="red",
        label="Exclusivo",
    )
    # plt.hist(
    #     energias,
    #     weights=contagens_I,
    #     bins=len(energias),
    #     align="mid",
    #     color="black",
    #     label="Inclusivo",
    # )
    # plt.hist(
    #     energias,
    #     weights=contagens_E,
    #     bins=len(energias),
    #     align="mid",
    #     color="red",
    #     label="Exclusivo",
    # )
    plt.xlabel(r"$E_{CM}$ (MeV)")
    # plt.xlim(left=2, right=35)
    plt.xlim(left=0.6, right=6.5)
    plt.ylabel("$\sigma$ (mb)")
    # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(250))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(50))
    # plt.legend(framealpha=1.0, edgecolor="black")
    # plt.colorbar()
    plt.tight_layout()
    # plt.savefig("figs/dist_angs/dist_ang_contagens", dpi=600, bbox_inches="tight")
    plt.show()


def f4():
    """Get python.h file path."""
    a = os.path.dirname(os.__file__)
    a = os.path.join(a, "..", "include")
    print(os.listdir(a))
    print(a)


if __name__ == "__main__":
    # f3()
    f5()
