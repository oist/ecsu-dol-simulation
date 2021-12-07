import numpy as np
from itertools import chain, combinations
from sklearn import preprocessing

def interpret_observed_effect_size(effectSize, whichOne):
    if whichOne == 1: #####  Eta^2 OR Epsilon^2
        if effectSize <= 0.01:					
            return 'Very Small Effect'
        elif 0.01 < effectSize < 0.06:					
            return 'Small Effect'
        elif 0.06 <= effectSize < 0.14:					
            return 'Medium Effect'
        elif effectSize >= 0.14:
            return 'Large Effect'
    elif whichOne == 2:				
        if effectSize < 0.1:					
            return 'Very Small Effect'
        elif 0.01 <= effectSize < 0.3:		
            return 'Small Effect'
        elif 0.3 <= effectSize < 0.5:					
            return 'Medium Effect'
        elif effectSize >= 0.5:
            return 'Large Effect'				

def show_descriptive_stats(data, whichOne):
    print('M-' + whichOne, ' = ', np.mean(data), ' SD-' + whichOne, ' = ', np.std(data), '  Mdn-' + whichOne, ' = ', np.median(data), \
        '  CI_95%-' + whichOne + ' = ', [np.percentile(data, 2.5), np.percentile(data, 97.5)])


def normalize_data(matrix, norm_type):
    """[summary]

    Args:
        matrix ([type]): input matrix
        norm_type ([type]): 0 : Use Orginal Data   1 : Z-Score Normalization   2 : [0 .. 1] Scaling			

    Returns:
        [type]: [description]
    """
    if norm_type == 0:
        return matrix
    scaler = \
        preprocessing.StandardScaler().fit(matrix) if norm_type == 1 \
        else preprocessing.MinMaxScaler().fit(matrix)
    return scaler.transform(matrix)

def powerset_idx(num_elements, remove_empty=False):
    elements = list(range(num_elements))
    min_size = 1 if remove_empty else 0
    result = chain.from_iterable(combinations(elements, r) for r in range(min_size, num_elements+1))
    return list(result)

def test_powerset_idx():
    num = 5
    remove_empty = True
    ps = powerset_idx(num, remove_empty)
    size = len(ps)
    exp_size = 2 ** num    
    if remove_empty:
        exp_size -= 1
    assert exp_size == size
    print(ps)
    print(size)

if __name__ == "__main__":
    test_powerset_idx