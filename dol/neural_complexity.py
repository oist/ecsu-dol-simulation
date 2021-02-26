import numpy as np
import os

def bipartition(collection):
    assert len(collection) > 1
    if len(collection) == 2:
        yield [ [collection[0]], [collection[1]] ]
        return
    first = collection[0]
    rest = collection[1:]
    # put `first` in its own subset 
    yield [ [ first ] ] + [ rest ]        
    for smaller in bipartition(rest):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]    


def sort_bipartition_by_size(bipart_list):
    for bipart in bipart_list:
        bipart.sort(key = len)


def get_entropy_using_coveriance(data):
    assert data.ndim == 2
    num_nodes = len(data)
    cov_matrix = np.cov(data) # num_nodes x num_nodes    
    if num_nodes == 1: # single row
        cov_det = float(cov_matrix) # cov_matrix is a 0-dim array
    else:
        cov_det = np.linalg.det(cov_matrix)
    log_arg = ((2 * np.pi * np.e) ** num_nodes) * cov_det
    assert log_arg > 0
    entropy = 0.5 * np.log( log_arg )
    return entropy
    

def test_bipartitions(n):
    l = list(range(1,n+1))
    bipart_list = list(bipartition(l))
    sort_bipartition_by_size(bipart_list) # sort by size of subpartitions
    num = len(bipart_list)
    assert num == (2 ** n - 2) / 2
    print('Bipartition of {} ({}):'.format(l, num))
    print('\n'.join(str(x) for x in bipart_list))

def compute_neural_complexity(data, rs=None):
    assert data.ndim == 2
    assert data.shape[0] < data.shape[1] # few rows, many columns
    if rs is not None:
        # add noise
        noise = rs.normal(0, 1e-8, data.shape)
        data = data + noise
    num_nodes, _ = data.shape    
    h_AB = get_entropy_using_coveriance(data)
    # print('entropy AB: ', h_AB)
    index_list = list(range(num_nodes))
    bipart_list = list(bipartition(index_list)) # all possible bipartitions of indexes
    sort_bipartition_by_size(bipart_list) # sort by size of bipartitions sets
    
    # keep track of mi values of each group indexed by k-1
    # (where k is the size of smallest bipartition set)
    mi_group_size = int(num_nodes/2)
    mi_groups_val = np.zeros(mi_group_size) 
    mi_groups_len = np.zeros(mi_group_size) 
    for bipart in bipart_list:
        A = data[bipart[0], :] # rows specified in indexes of first bipartition set
        B = data[bipart[1], :] # rows specified in indexes of second bipartition set
        k = len(A)
        h_A = get_entropy_using_coveriance(A)
        h_B = get_entropy_using_coveriance(B)
        mi = h_A + h_B - h_AB
        mi_groups_val[k-1] += mi
        mi_groups_len[k-1] += 1
    neural_complexity = np.sum(mi_groups_val / mi_groups_len) # sum over averaged values
    return neural_complexity


def test_complexity_random_uniform(num_nodes, num_data_points, seed):
    rs = np.random.RandomState(seed)
    data = rs.uniform(-1,1,size=(num_nodes, num_data_points))
    nc = compute_neural_complexity(data, rs)
    print('Nerual Complexity (Random Uniform)', nc)


def test_complexity_random_gaussian(num_nodes, num_data_points, seed):
    rs = np.random.RandomState(seed)
    data = rs.normal(size=(num_nodes, num_data_points))
    nc = compute_neural_complexity(data, rs)
    print('Nerual Complexity (Random Normal Gaussian)', nc)


def test_complexity_constant(num_nodes, num_data_points, seed):
    rs = np.random.RandomState(seed)
    data = np.ones((num_nodes, num_data_points))
    nc = compute_neural_complexity(data, rs)
    print('Nerual Complexity (Constat-ones)', nc)


def generate_correlated_data(num_data_points, cov, rs):
    # see https://quantcorner.wordpress.com/2018/02/09/generation-of-correlated-random-numbers-using-python/
    from scipy.linalg import cholesky
    from scipy.stats import pearsonr
    
    # Compute the (upper) Cholesky decomposition matrix
    corr_mat= np.array(
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
    assert num_nodes%2 == 0

    rs = np.random.RandomState(seed)
    num_pairs = int(num_nodes/2)
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
    A = np.expand_dims(AB[0,:],0) # first row as 2 dim array
    B = np.expand_dims(AB[1,:],0) # second row as 2 dim array
    h_A = get_entropy_using_coveriance(A)
    h_B = get_entropy_using_coveriance(B)
    h_AB = get_entropy_using_coveriance(AB)    
    return h_A + h_B - h_AB


def test_mutual_information(seed):
    num_data_points = 500
    
    rs = np.random.RandomState(seed)    
    mi_random_uniform = compute_mutual_information(
        rs.uniform(-1,1,size=(2, num_data_points))
    )

    rs = np.random.RandomState(seed)    
    mi_random_gaussian = compute_mutual_information(
        rs.normal(size=(2, num_data_points))
    )
    
    rs = np.random.RandomState(seed)    
    mi_random_gaussian = compute_mutual_information(
        rs.normal(size=(2, num_data_points))
    )

    rs = np.random.RandomState(seed)    
    mi_constant = compute_mutual_information(
        np.ones((2, num_data_points)) +
        rs.normal(0, 1e-15, size=(2, num_data_points))
    )

    rs = np.random.RandomState(seed)    
    mi_correlated = compute_mutual_information(
        generate_correlated_data(num_data_points, cov=0.9, rs=rs)
    )
    print('MI random uniform:', mi_random_uniform)
    print('MI random gaussian:', mi_random_gaussian)
    print('MI constant:', mi_constant)
    print('MI correlated:', mi_correlated)


def test_complexity(num_nodes, seed):
    num_data_points=500
    
    test_complexity_random_uniform(num_nodes, num_data_points, seed)
    
    test_complexity_random_gaussian(num_nodes, num_data_points, seed)

    test_complexity_constant(num_nodes, num_data_points, seed)

    test_complexity_correlated_data(num_nodes, num_data_points, cov=0.9, seed=seed)

def test():
    # test_bipartitions(n=4)
    # generate_correlated_data(num_data_points=500, cov=0.9, seed=0, rs = np.random.RandomState())
    # test_mutual_information(seed=0)
    test_complexity(num_nodes=6,seed=0)



if __name__ == "__main__":    
    test()    
