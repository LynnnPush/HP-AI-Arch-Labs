import numpy as np
import matplotlib.pyplot as plt


def plot_xs(T: list[float], Xs: list[list[list[float]]], speed: float, save_path: str | None = None):
    T_np = np.array(T)
    Xs_np = np.array(Xs)

    plt.plot(T_np[::5], Xs_np[:, 0, ::5].T, "k", alpha=0.3)
    plt.title(f"Speed = {speed} mm/ms")
    plt.xlabel("time (ms)")
    plt.ylabel("X(t)")
    plt.xlim(0, T[-1])
    plt.grid(axis='x')

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_delay_hist(
    D: list[list[float]], W: list[list[float]], speed: float, save_path: str | None = None
):
    D_np = np.array(D)
    W_np = np.array(W)
    # T_np = np.array(T)

    delays = (D_np[W_np != 0] / speed).flatten()

    plt.hist(delays, bins=100)
    # plt.xlim(0, T_np[-1])
    plt.title(f"Speed = {speed} mm/ms")
    plt.xlabel("delay (ms)")
    plt.ylabel("# delay")

    if save_path:
        plt.savefig(save_path)

    plt.show()
