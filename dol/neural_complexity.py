"""
Implements TSE complexity.
"""
import numpy as np
from numpy.random import RandomState

def bipartition(collection):
    assert len(collection) > 1
    if len(collection) == 2:
        yield [[collection[0]], [collection[1]]]
        return
    first = collection[0]
    rest = collection[1:]
    # put `first` in its own subset 
    yield [[first]] + [rest]
    for smaller in bipartition(rest):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]


def sort_bipartition_by_size(bipart_list):
    for bipart in bipart_list:
        bipart.sort(key=len)


def get_entropy_from_cov_det(cov_det, num_nodes):
    log_arg = ((2 * np.pi * np.e) ** num_nodes) * cov_det
    assert log_arg > 0
    entropy = 0.5 * np.log(log_arg)
    return entropy


def get_entropy_using_coveriance(data):
    assert data.ndim == 2
    num_nodes = len(data)
    if num_nodes == 1: # single row
        cov_det = np.var(data)
    else:
        cov_matrix = np.cov(data)  # num_nodes x num_nodes
        cov_det = np.linalg.det(cov_matrix)
    return get_entropy_from_cov_det(cov_det, num_nodes)


def test_bipartitions(n):
    l = list(range(1, n + 1))
    bipart_list = list(bipartition(l))
    sort_bipartition_by_size(bipart_list)  # sort by size of subpartitions
    num = len(bipart_list)
    assert num == (2 ** n - 2) / 2
    print('Bipartition of {} ({}):'.format(l, num))
    print('\n'.join(str(x) for x in bipart_list))


def compute_neural_complexity_n1n2_single(data, n1_idx, n2_idx, rs=None):    
    assert data.ndim == 2
    assert data.shape[0] < data.shape[1]  # few rows, many columns
    if rs is not None:
        # add noise
        noise = rs.normal(0, 1e-5, data.shape)
        data = data + noise
    num_nodes, _ = data.shape

    # only consider 2 partitionings of the nodes in data
    # p1: n1, X - n1
    # p2: n2, X - n2
    nodes_idx = list(range(num_nodes))
    nodes_idx_not_n1 = [i for i in nodes_idx if i != n1_idx]
    nodes_idx_not_n2 = [i for i in nodes_idx if i != n2_idx]
    bipart_list = [        
        [[n1_idx], nodes_idx_not_n1],
        [[n2_idx], nodes_idx_not_n2],
    ]

    # entrpy of the whole system
    h_AB = get_entropy_using_coveriance(data)

    # print('entropy AB: ', h_AB)
    index_list = list(range(num_nodes))
    bipart_list = list(bipartition(index_list))  # all possible bipartitions of indexes
    sort_bipartition_by_size(bipart_list)  # sort by size of bipartitions sets

    mi_list = []
    # compute the two MI
    for bipart in bipart_list:
        A = data[bipart[0]]  # rows specified in indexes of first bipartition set
        B = data[bipart[1]]  # rows specified in indexes of second bipartition set        
        h_A = get_entropy_using_coveriance(A)
        h_B = get_entropy_using_coveriance(B)
        mi = h_A + h_B - h_AB
        mi_list.append(mi)
        
    neural_complexity = np.mean(mi_list)  # make the average of the resulting MI    
    return neural_complexity


def compute_neural_complexity_n1n2_combined(data1, data2, n1_idx, n2_idx, rs=None):
    # data1 and data2 are the data from ag1 and ag2
    assert data1.ndim == data2.ndim == 2
    assert data1.shape == data2.shape # same shape
    assert data1.shape[0] < data1.shape[1]  # few rows, many columns
    num_nodes, _ = data1.shape
    if rs is not None:
        # add noise
        noise1 = rs.normal(0, 1e-5, data1.shape)
        noise2 = rs.normal(0, 1e-5, data2.shape)
        data1 = data1 + noise1        
        data2 = data1 + noise2

    # consider the following partionings
    # p1 = (n1), (n2, s1, s2, m1, m2)
    # p2 = (n2), (n1, s1, s2, m1, m2)
    nodes_idx = list(range(num_nodes))
    nodes_idx_not_n1 = [i for i in nodes_idx if i != n1_idx]
    nodes_idx_not_n2 = [i for i in nodes_idx if i != n2_idx]
    
    # compute cov
    var_n1 = np.var(data1[n1_idx]) + np.var(data2[n1_idx]) # cov_n1 = var(n1_a1) + var(n1_a2)
    var_n2 = np.var(data1[n2_idx]) + np.var(data2[n2_idx]) # cov_n2 = var(n2_a1) + var(n2_a2)
    cov_not_n1 = np.cov(data1[nodes_idx_not_n1]) + np.cov(data2[nodes_idx_not_n1])
    cov_not_n2 = np.cov(data1[nodes_idx_not_n2]) + np.cov(data2[nodes_idx_not_n2])
    cov_Y = np.cov(data1) + np.cov(data2) # covariance of the whole system        
    # determinants of cov 
    cov_det_not_n1 = np.linalg.det(cov_not_n1)
    cov_det_not_n2 = np.linalg.det(cov_not_n2)
    cov_det_Y = np.linalg.det(cov_Y)
    # entropies based on cov_det
    h_n1 = get_entropy_from_cov_det(var_n1, 1)
    h_n2 = get_entropy_from_cov_det(var_n2, 1)    
    h_not_n1 = get_entropy_from_cov_det(cov_det_not_n1, num_nodes-1)
    h_not_n2 = get_entropy_from_cov_det(cov_det_not_n2, num_nodes-1)
    h_Y = get_entropy_from_cov_det(cov_det_Y, num_nodes)
    # mutual informations
    mi_p1 = h_n1 + h_not_n1 - h_Y # MI first partition
    mi_p2 = h_n2 + h_not_n2 - h_Y # MI second partition
    
    # neural complexity
    C_Y = np.mean([mi_p1, mi_p2]) 

    return C_Y


def compute_neural_complexity(data, rs=None):
    assert data.ndim == 2
    assert data.shape[0] < data.shape[1]  # few rows, many columns
    if rs is not None:
        # add noise
        noise = rs.normal(0, 1e-5, data.shape)
        data = data + noise
    num_nodes, _ = data.shape
    h_AB = get_entropy_using_coveriance(data)
    # print('entropy AB: ', h_AB)
    index_list = list(range(num_nodes))
    bipart_list = list(bipartition(index_list))  # all possible bipartitions of indexes
    sort_bipartition_by_size(bipart_list)  # sort by size of bipartitions sets

    # keep track of mi values of each group indexed by k-1
    # (where k is the size of smallest bipartition set)
    mi_group_size = int(num_nodes / 2)
    mi_groups_val = np.zeros(mi_group_size)
    mi_groups_len = np.zeros(mi_group_size)
    for bipart in bipart_list:
        A = data[bipart[0]]  # rows specified in indexes of first bipartition set
        B = data[bipart[1]]  # rows specified in indexes of second bipartition set
        k = len(A)
        h_A = get_entropy_using_coveriance(A)
        h_B = get_entropy_using_coveriance(B)
        mi = h_A + h_B - h_AB
        mi_groups_val[k - 1] += mi
        mi_groups_len[k - 1] += 1
    neural_complexity = np.sum(mi_groups_val / mi_groups_len)  # sum over averaged values
    return neural_complexity


def test_complexity_random_uniform(num_nodes, num_data_points, seed):
    rs = RandomState(seed)
    data = rs.uniform(-1, 1, size=(num_nodes, num_data_points))
    nc = compute_neural_complexity(data, rs)
    print('Nerual Complexity (Random Uniform)', nc)


def test_complexity_random_gaussian(num_nodes, num_data_points, seed):
    rs = RandomState(seed)
    data = rs.normal(size=(num_nodes, num_data_points))
    nc = compute_neural_complexity(data, rs)
    print('Nerual Complexity (Random Normal Gaussian)', nc)


def test_complexity_constant(num_nodes, num_data_points, seed):
    rs = RandomState(seed)
    data = np.ones((num_nodes, num_data_points))
    nc = compute_neural_complexity(data, rs)
    print('Nerual Complexity (Constat-ones)', nc)


def generate_correlated_data(num_data_points, cov, rs):
    # see https://quantcorner.wordpress.com/2018/02/09/generation-of-correlated-random-numbers-using-python/
    from scipy.linalg import cholesky
    from scipy.stats import pearsonr

    # Compute the (upper) Cholesky decomposition matrix
    corr_mat = np.array(
        [
            [1.0, cov],
            [cov, 1.0]
        ]
    )

    # Compute the (upper) Cholesky decomposition matrix
    upper_chol = cholesky(corr_mat)

    # Generate 3 series of normally distributed (Gaussian) numbers
    rnd = rs.normal(0.0, 1.0, size=(num_data_points, 2))

    result = rnd @ upper_chol
    # corr_0_1 , _ = pearsonr(result[:,0], result[:,1])
    # print(result.shape)
    # print(corr_0_1)

    return np.transpose(result)


def test_complexity_correlated_data(num_nodes, num_data_points, cov, seed):
    assert num_nodes % 2 == 0

    rs = RandomState(seed)
    num_pairs = int(num_nodes / 2)
    # stuck together pairs of coorelated rows
    data = np.row_stack(
        [
            generate_correlated_data(num_data_points, cov, rs)
            for _ in range(num_pairs)
        ]
    )
    assert data.shape == (num_nodes, num_data_points)
    nc = compute_neural_complexity(data, rs)
    print('Nerual Complexity (Correlated)', nc)


def compute_mutual_information(AB):
    assert AB.ndim == 2
    A = np.expand_dims(AB[0, :], 0)  # first row as 2 dim array
    B = np.expand_dims(AB[1, :], 0)  # second row as 2 dim array
    h_A = get_entropy_using_coveriance(A)
    h_B = get_entropy_using_coveriance(B)
    h_AB = get_entropy_using_coveriance(AB)
    return h_A + h_B - h_AB


def test_mutual_information(seed):
    num_data_points = 500

    rs = RandomState(seed)
    mi_random_uniform = compute_mutual_information(
        rs.uniform(-1, 1, size=(2, num_data_points))
    )

    rs = RandomState(seed)
    mi_random_gaussian = compute_mutual_information(
        rs.normal(size=(2, num_data_points))
    )

    rs = RandomState(seed)
    mi_random_gaussian = compute_mutual_information(
        rs.normal(size=(2, num_data_points))
    )

    rs = RandomState(seed)
    mi_constant = compute_mutual_information(
        np.ones((2, num_data_points)) +
        rs.normal(0, 1e-15, size=(2, num_data_points))
    )

    rs = RandomState(seed)
    mi_correlated = compute_mutual_information(
        generate_correlated_data(num_data_points, cov=0.9, rs=rs)
    )
    print('MI random uniform:', mi_random_uniform)
    print('MI random gaussian:', mi_random_gaussian)
    print('MI constant:', mi_constant)
    print('MI correlated:', mi_correlated)


def test_complexity(num_nodes, seed):
    num_data_points = 500

    test_complexity_random_uniform(num_nodes, num_data_points, seed)

    test_complexity_random_gaussian(num_nodes, num_data_points, seed)

    test_complexity_constant(num_nodes, num_data_points, seed)

    test_complexity_correlated_data(num_nodes, num_data_points, cov=0.9, seed=seed)


def test():
    # test_bipartitions(n=4)
    # generate_correlated_data(num_data_points=500, cov=0.9, seed=0, rs = RandomState())
    # test_mutual_information(seed=0)
    test_complexity(num_nodes=6, seed=0)
    # TODO: test compute_neural_complexity_n1n2_single
    # TODO: test compute_neural_complexity_n1n2_combined


if __name__ == "__main__":
    test()
