import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sb
from numpy.random import RandomState

def heat_map(data, labels, title, xlabel, ylabel, output_file=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # pylint: disable=maybe-no-member
    cax = ax.matshow(data, cmap = cm.Spectral_r, interpolation = 'nearest')
    fig.colorbar(cax)

    xaxis = np.arange(len(labels))
    ax.set_xticks(xaxis)
    ax.set_yticks(xaxis)
    ax.set_xticklabels(labels, rotation = 90)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)

    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()

def box_plot(data, labels, title, ylabel, output_file=None, random_seed=0):
    np.random.seed(random_seed) # reproducibility		
    plt.figure()
    meanprops={
        "marker" : "o",
        "markerfacecolor" : "white", 
        "markeredgecolor" : "black",
        "markersize" : "10"
    }
    sb.boxplot(data = data, showmeans = True, meanprops=meanprops, showfliers=False)
    sb.stripplot(color='black', data = data)
    plt.xticks(range(0, len(labels)), labels, rotation = 0)
    plt.title(title)
    plt.ylabel(ylabel, fontsize = 25)

    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()

def test_box_plot():
    rs = RandomState(123)
    labels_data = {
        'one': rs.random_sample(100),
        'two': rs.random_sample(100),
        'three': rs.random_sample(100),
    }
    labels = list(labels_data.keys())
    data = list(labels_data.values())
    box_plot(data, labels, title='box plot', ylabel='')

if __name__ == "__main__":
    test_box_plot()