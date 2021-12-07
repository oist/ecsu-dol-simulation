# dyadic integrated information
import numpy as np
import jpype as jp
from numpy.random import RandomState
from dol.info_analysis import info_utils, plots, infodynamics
from dol.info_analysis.infodynamics import compute_mi, compute_cond_mi

class DII:

    def __init__(self, matrix_a, matrix_b, matrix_conditioning, norm_type=0):
        """[summary]

        Args:
            matrix_a (ndarray): time_steps x num_variables (>=2)
            matrix_b (ndarray): time_steps x num_variables (>=2)
            norm_type (int): normalization (default to 0 for no normalization)            
        """        
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b
        self.conditioning_matrix = np.expand_dims(matrix_conditioning, axis = 1) # make 2 dim for infodynamics computation
        
        self.time_steps, self.num_columns = self.matrix_a.shape
        self.norm_type =norm_type

        # check parameters
        self.__check_params()
        
        # matrix normalization
        matrix_a = info_utils.normalize_data(self.matrix_a, self.norm_type)
        matrix_b = info_utils.normalize_data(self.matrix_b, self.norm_type)


    def __check_params(self):
        assert self.matrix_a.shape == self.matrix_b.shape, \
            "The two input matrices must have same shape"
        
        assert self.matrix_a.shape[0] == self.conditioning_matrix.shape[0], \
            "The conditioning matrix should have same number of elements as the other matrices"

        assert self.matrix_a.ndim == 2, \
            "The two input matrices must have 2 dimensions"
                
        assert self.time_steps > self.num_columns, \
            "First dimension is time series (must be greater than second dim)"


    def compute_dii_powerset(self, plot=False):
        """compute dyadic integrated information

        Args:
            plot (bool): if to plot results
        """

        # computer power set of each matrix columns (except for empty set)
        power_set_idx = info_utils.powerset_idx(self.num_columns, remove_empty=True)
        heat_map_size = len(power_set_idx)

        COND_MI_matrix = np.zeros((heat_map_size, heat_map_size))
        MI_matrix = np.zeros((heat_map_size, heat_map_size))

        # MI_matrix and COND_MI_matrix are computed as powerset partitioning (over the columns of agent A and agent B matrix)
        # overall_MI is a single value multivariate MI computed from the full matrix of agent A and anget B 

        for i, ps_a in enumerate(power_set_idx):
            for j, ps_b in enumerate(power_set_idx):            
                sub_matrix_a = self.matrix_a[:,ps_a] # column indexes specified in ps_a
                sub_matrix_b = self.matrix_b[:,ps_b] # column indexes specified in ps_b
                MI_matrix[i][j] = compute_mi(sub_matrix_a, sub_matrix_b)
                COND_MI_matrix[i][j] = compute_cond_mi(sub_matrix_a, sub_matrix_b, self.conditioning_matrix)

        if plot:
            matrices = [MI_matrix, COND_MI_matrix]
            titles = ['MI heat map', 'Cond MI heat map']
            for matrix, title in  zip(matrices, titles):
                labels = [','.join([str(i+1) for i in idx]) for idx in power_set_idx]
                plots.heat_map(
                    matrix, 
                    labels, 
                    title=title,
                    xlabel='A',
                    ylabel='B'
                )        

        return MI_matrix, COND_MI_matrix

    def compute_dii_overall(self):
        """compute dyadic integrated information

        Args:
            plot (bool): if to plot results
        """

        # computer power set of each matrix columns (except for empty set)
        power_set_idx = info_utils.powerset_idx(self.num_columns, remove_empty=True)
        heat_map_size = len(power_set_idx)

        overall_MI = compute_mi(self.matrix_a, self.matrix_b)
        overall_COND_MI = compute_cond_mi(self.matrix_a, self.matrix_b, self.conditioning_matrix)

        return overall_MI, overall_COND_MI



def test():

    rs = RandomState(123)
    shape = (1000, 3)
    matrix_a = rs.random_sample(shape)
    matrix_b = rs.random_sample(shape)
    matrix_conditioning = rs.random_sample((shape[0], 1))
    dii = DII(matrix_a, matrix_b, matrix_conditioning, norm_type=0)
    mi_matrix, cond_mi_matrix, overall_mi = dii.compute_dii(plot=True)
    
    mi_mean = mi_matrix.mean()
    mi_mean_a = mi_matrix.mean(axis=0) # averaging across all values of B
    mi_mean_b = mi_matrix.mean(axis=1) # averaging across all values of A
    print('MI Mean:', mi_mean)
    print('MI Mean A:', mi_mean_a.tolist())
    print('MI Mean B:', mi_mean_b.tolist())

    cond_mi_mean = cond_mi_matrix.mean()
    cond_mi_mean_a = cond_mi_matrix.mean(axis=0) # averaging across all values of B
    cond_mi_mean_b = cond_mi_matrix.mean(axis=1) # averaging across all values of A
    print('Cond MI Mean:', cond_mi_mean)
    print('Cond MI Mean A:', cond_mi_mean_a.tolist())
    print('Cond MI Mean B:', cond_mi_mean_b.tolist())

    synergy_powerset = cond_mi_mean - mi_mean
    print('Synergy powerset (cond_mi_mean - mi_mean):', synergy_powerset)

    print('Overall MI (non power set)', overall_mi)

    synergy_non_powerset = cond_mi_mean - overall_mi
    print('Synergy non-powerset (cond_mi_mean - overall_mi):', synergy_non_powerset)


if __name__ == "__main__":
    test()    