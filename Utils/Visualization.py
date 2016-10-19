import numpy as np
import matplotlib.pyplot as plotter


def new_figure(num, epoch_in_retry, max_y):
    x = np.arange(0, epoch_in_retry, 1)
    yv = np.full(epoch_in_retry, -1.1)
    yt = np.full(epoch_in_retry, -1.1)
    fig = plotter.figure(num)
    fig.set_size_inches(1920, 1080)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, epoch_in_retry)
    ax.set_ylim(0, max_y)
    ax.set_xlabel('epoch')
    ax.set_ylabel('error')
    ax.grid(True)
    line = ax.plot(x, yv, 'r.', x, yt, 'b.')
    fig.show(False)
    fig.canvas.draw()
    return line, fig


def update_figure(plot, axes, x, yv, yt):
    new_data = axes[0].get_ydata()
    new_data[x] = yv
    axes[0].set_ydata(new_data)
    new_data = axes[1].get_ydata()
    new_data[x] = yt
    axes[1].set_ydata(new_data)
    plot.canvas.draw()


def save_to_file(plot, filename):
    plot.savefig(filename)
    plotter.close(plot)
