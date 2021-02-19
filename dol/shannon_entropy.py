import numpy as np
from collections import defaultdict

BINS = 100

def get_shannon_entropy_1d(data_1d, min_v=0., max_v=100.):
    num_data_points = len(data_1d)
    histo, _ = np.histogram(
        data_1d,
        bins=BINS,
        range=[min_v, max_v]
    )

    # print(histo)
    with np.errstate(divide='ignore'):
        histo_prob = histo / num_data_points
        histo_neg_prob = np.negative(histo_prob)
        hist_log_prob = np.log2(histo_prob)
    with np.errstate(invalid='ignore'):
        histo_neg_prob_log_prob = np.multiply(histo_neg_prob, hist_log_prob)
    entropy = np.nansum(histo_neg_prob_log_prob)

    norm_factor = min(BINS, num_data_points)
    distance_shannon_entropy = entropy / np.log2(norm_factor)
    return distance_shannon_entropy


def get_shannon_entropy_2d(brain_output, min_v=0., max_v=1.):
    num_data_points = len(brain_output)
    histo, _, _ = np.histogram2d(
        brain_output[:, 0],
        brain_output[:, 1],
        bins=[BINS, BINS],
        range=[[min_v, max_v], [min_v, max_v]],
    )
    # print(histo)
    with np.errstate(divide='ignore'):
        histo_prob = histo / num_data_points
        histo_neg_prob = np.negative(histo_prob)
        hist_log_prob = np.log2(histo_prob)
    with np.errstate(invalid='ignore'):
        histo_neg_prob_log_prob = np.multiply(histo_neg_prob, hist_log_prob)
    entropy = np.nansum(histo_neg_prob_log_prob)
    norm_entropy = entropy / np.log2(num_data_points)
    return norm_entropy

def get_shannon_entropy_dd(data, min_v=0., max_v=1.):
    num_data_points = len(data)
    dimensions = data.shape[1]
    histo, _ = np.histogramdd(
        data,
        bins = BINS,
        range =[(min_v, max_v)] * dimensions,
    )
    # print(histo)
    with np.errstate(divide='ignore'):
        histo_prob = histo / num_data_points
        histo_neg_prob = np.negative(histo_prob)
        hist_log_prob = np.log2(histo_prob)
    with np.errstate(invalid='ignore'):
        histo_neg_prob_log_prob = np.multiply(histo_neg_prob, hist_log_prob)
    entropy = np.nansum(histo_neg_prob_log_prob)
    norm_entropy = entropy / np.log2(num_data_points)
    return norm_entropy

def get_shannon_entropy_dd_simplified(data, min_v=0., max_v=1.):
    num_data_points = len(data)
    dimensions = data.shape[1]
    binning_space = np.linspace(min_v, max_v, BINS)

    histo_columns = []
    for d in range(dimensions):                
        column_data = data[:,[d]]
        digitized_column_data = np.digitize(column_data, binning_space)
        histo_columns.append(digitized_column_data)
    
    dim_bins = np.column_stack(histo_columns)
    # shape is still (num_data_points, dimensions)
    # i,j is an index between 0 and BINS-1 corresponding to the binning of data[i][j]
    # print(dim_bins.shape)
    assert dim_bins.shape == (num_data_points, dimensions)
    bin_dict = defaultdict(float)
    for i in range(num_data_points):
        binned_row = tuple(dim_bins[i,:])
        bin_dict[binned_row] += 1
    histo = np.array(list(bin_dict.values()))
    histo_prob = histo / num_data_points
    histo_neg_prob = np.negative(histo_prob)
    hist_log_prob = np.log2(histo_prob)
    histo_neg_prob_log_prob = np.multiply(histo_neg_prob, hist_log_prob)
    entropy = np.nansum(histo_neg_prob_log_prob)
    norm_entropy = entropy / np.log2(num_data_points)
    return norm_entropy


    # print(histo)
    # with np.errstate(divide='ignore'):
    #     histo_prob = histo / num_data_points
    #     histo_neg_prob = np.negative(histo_prob)
    #     hist_log_prob = np.log2(histo_prob)
    # with np.errstate(invalid='ignore'):
    #     histo_neg_prob_log_prob = np.multiply(histo_neg_prob, hist_log_prob)
    # entropy = np.nansum(histo_neg_prob_log_prob)
    # norm_entropy = entropy / np.log2(BINS ** dimensions)
    # return norm_entropy

def test_1d():
    # a = np.array([1.1, 2.2, 3.3, 4.4])
    a = np.random.random(1000)
    # print(a)
    distance_entropy = get_shannon_entropy_1d(a, min_v=0., max_v=1.)
    print(distance_entropy)

def test_2d():
    a = np.array([[.1, .2, .3, .4],[.1, .2, .3, .4]])
    a = np.transpose(a)
    print(a)
    norm_entropy = get_shannon_entropy_2d(a)
    print(norm_entropy)

def test_dd():
    a = np.array([
        [.0, .0, .0]
    ])
    print(a.shape[1])
    norm_entropy = get_shannon_entropy_dd(a)
    print(norm_entropy)

def test_dd_simple():
    data = np.random.random((100,5))
    # entropy_dd = get_shannon_entropy_dd(data)
    entropy_dd_simple = get_shannon_entropy_dd_simplified(data)
    # print('entropy_dd: {}'.format(entropy_dd))
    print('entropy_dd_simple: {}'.format(entropy_dd_simple))

def shannon_plot(dim=2):
    import matplotlib.pyplot as plt
    import math
    data_points = 2000
    bins_per_dim = 100    
    X = list(range(1,data_points+1))
    
    # original implementation by candadai (wrong)
    #Y = [- x/data_points * math.log2(1/data_points) / math.log2(bins_per_dim ** dim) for x in X] 
    
    # new implementation (correct)
    Y = [- x/data_points * math.log2(1/data_points) / math.log2(data_points) for x in X] 
    plt.plot(X, Y)
    
    print(Y[1999])
    plt.show()
    

if __name__ == "__main__":
    test_1d()    
    # test_2d()
    # test_dd()
    # test_dd_simple()
    # shannon_plot(1)