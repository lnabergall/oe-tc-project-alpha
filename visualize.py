
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from storage import *
from system import ParticleSystem as System


def generate_movie(states, system, frame_interval):
    fig, ax = plt.subplots(figsize=(10, 10))
    scat = ax.scatter([], [], s=10, c=[], cmap="coolwarm", vmin=-1, vmax=1)
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


def save_movie(ani, config_name, time, frame_interval):
    file_name = get_foldername(config_name, time) + "/movie.mp4"
    Writer = animation.FFMpegWriter
    ani.save(file_name, writer=Writer(fps=1000//frame_interval, bitrate=8000), dpi=300)


def create_movie(config_name, time, frame_interval=200):
    time = string_to_datetime(time)
    config = load_config(config_name, time)
    system = System(**config)
    states = load_states(config_name, time)
    ani = generate_movie(states, system, frame_interval)
    save_movie(ani, config_name, time, frame_interval)
    plt.show()
