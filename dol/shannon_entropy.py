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
    assert data.ndim == 2
    num_data_points = len(data) 
    # number of samples must be 1 more than number of bins - so there are "bins" intervals
    binning_space = np.linspace(min_v, max_v, num=BINS+1)    

    data_binned = np.digitize(data, binning_space)
    data_binned = np.clip(data_binned, a_min=None, a_max=BINS)
    # we need to clip to make sure that right-most edge (1) goes to last bin, 
    # see https://github.com/numpy/numpy/issues/4217

    # shape is the same as original data
    # i,j is an index between 1 and BINS corresponding to the binning of data[i][j]
    # print(data_binned.shape)
    assert data_binned.shape == data.shape

    bin_dict = defaultdict(float)
    for i in range(num_data_points):
        binned_row = tuple(data_binned[i])
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
    a = np.random.random(1000) # pylint: disable-msg=no-member
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
    data = np.random.random((100,5)) # pylint: disable-msg=no-member
    entropy_dd_simple = get_shannon_entropy_dd_simplified(data)
    print('entropy_dd_simple: {}'.format(entropy_dd_simple))
    print('---')
    data = np.random.random((10000,2)) # pylint: disable-msg=no-member
    entropy_2d = get_shannon_entropy_2d(data)
    entropy_dd_simple = get_shannon_entropy_dd_simplified(data)
    print('entropy_2d: {}'.format(entropy_2d))
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

def test_binning():
    from collections import Counter
    min_v, max_v = 0., 1.
    num_bins = 10
    # number of samples must be 1 more than number of bins - so there are "bins" intervals
    binning_space = np.linspace(min_v, max_v, num=num_bins+1) 

    # data = np.random.random((100,2)) # pylint: disable-msg=no-member
    # data = np.ones((100,2))
    data = np.zeros((100,2))

    binned_data = np.digitize(data, binning_space)
    # we need to clip to make sure that right-most edge (1) goes to last bin, 
    # see https://github.com/numpy/numpy/issues/4217
    binned_data = np.clip(binned_data, a_min=None, a_max=num_bins)

    binned_data_counter = Counter([tuple(row) for row in binned_data])
    histo, _, _ = np.histogram2d(
        data[:, 0],
        data[:, 1],
        bins=binning_space,
        range=[[min_v, max_v], [min_v, max_v]]
    )
    for i in range(num_bins):
        for j in range(num_bins):
            histo_cell_count = histo[i,j]
            binned_data_count = binned_data_counter[i+1,j+1]
            assert histo_cell_count == binned_data_count
    # print(binned_data)
    # print(histo)
    entropy_2d = get_shannon_entropy_2d(data)
    entropy_dd_simple = get_shannon_entropy_dd_simplified(data)
    print('entropy 2d: {}'.format(entropy_2d))
    print('entropy dd simplified: {}'.format(entropy_dd_simple))

if __name__ == "__main__":
    # test_1d()    
    # test_2d()
    # test_dd()
    test_dd_simple()
    # shannon_plot(1)
    # test_binning()