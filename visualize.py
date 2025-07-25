
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from storage import *
from system import ParticleSystem as System
from evaluate import calculate_eval_data


def generate_movie(states, system, frame_interval):
    fig, ax = plt.subplots(figsize=(10, 10))
    s = max(5000 // system.k, 1)      # particle size
    scat = ax.scatter([], [], s=s, c=[], cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xlim(0, system.n)
    ax.set_ylim(0, system.n)
    ax.set_aspect("equal")
    ax.set_title("particle system")

    def init():
        scat.set_offsets(np.empty((0, 2)))
        scat.set_array(np.empty((0,)))
        return (scat,)

    def update(frame):
        positions = states.R[frame]
        scat.set_offsets(positions)
        scat.set_array(system.Q)
        return (scat,)

    ani = animation.FuncAnimation(
        fig, update, frames=states.step.size, init_func=init, blit=True, interval=frame_interval)
    return ani


def save_movie(ani, folder_name, frame_interval):
    file_name = folder_name + "/movie.mp4"
    Writer = animation.FFMpegWriter
    ani.save(file_name, writer=Writer(fps=1000//frame_interval, bitrate=8000), dpi=300)


def create_movie(states, config_name, time, folder_name, frame_interval):
    config = load_config(config_name, time)
    system = System(**config)
    ani = generate_movie(states, system, frame_interval)
    save_movie(ani, folder_name, frame_interval)


def plot_statistics(states, statistics, plot_name, folder_name, show_plot):
    file_name = folder_name + "/" + plot_name + "_plot.svg"

    plt.figure(figsize=(10, 4))
    for name, values in statistics:
        plt.plot(states.step, values, label=name)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(file_name)
    if show_plot:
        plt.show()


def generate_eval_data(states, folder_name, show_plot):
    total_stats, avg_stats, small_avg_stats, bs_stats = calculate_eval_data(states)

    plot_statistics(states, total_stats, "totals", folder_name, show_plot)
    plot_statistics(states, avg_stats, "avgs", folder_name, show_plot)
    plot_statistics(states, small_avg_stats, "small_avgs", folder_name, show_plot)
    plot_statistics(states, bs_stats, "bs_stats", folder_name, show_plot)


def produce_graphics(config_name, time, frame_interval=200, show_plot=False):
    folder_name = get_foldername(config_name, time)
    states = load_states(config_name, time)

    generate_eval_data(states, folder_name, show_plot)
    create_movie(states, config_name, time, folder_name, frame_interval)