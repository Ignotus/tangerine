#!/usr/bin/env python3
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

rcParams['legend.loc'] = 'best'


def plot(data, data_label, title, plot_name='plot.pdf'):
    figure, ax = plt.subplots()
    figure.set_size_inches((14, 10))

    width = 0.35

    ind = np.arange(len(data))
    ax.bar(ind + width, data, width, color='gray')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(data_label)

    delta = max(data) - min(data)
    plt.ylim([min(data) - delta * 0.01, max(data) + delta * 0.01])

    plt.ylabel('Value')
    plt.xlabel('Experiments')
    plt.title(title)
    plt.grid(True)

    with PdfPages(plot_name) as pdf:
        pdf.savefig(figure)
    plt.show()

if __name__ == '__main__':
    plot([10, -20, 15, 60], ['G1', 'G2', 'G3', 'G4'], 'RNN HSoftmax')

