#! /usr/bin/env python 3

from dol import analyze_results
import sys
import os
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sb
import jpype as jp
from sklearn import preprocessing
from joblib import Parallel, delayed
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import friedmanchisquare, ranksums, kruskal, spearmanr, pearsonr
import scipy.special as special
from dol.info_analysis import infodynamics
from dol.run_from_dir import run_simulation_from_dir
from numpy.polynomial.polynomial import polyfit
from dol.run_from_dir import run_simulation_from_dir
from numpy.random import RandomState
from itertools import combinations
from dol.info_analysis import info_utils


class InfoAnalysis:

    NORM_LABELS = {
        0: '',
        1: 'Z-Scored',
        2: '[0 ..1] Scaled'
    }	

    def __init__(self, sim_type_path, norm_type, 
        random_seed, num_cores=1, num_seeds_boostrapping=None, bootstrapping_runs=None,
        restrict_to_first_n_converged_seeds = None,
        output_dir=None, debug=True, plot=True, test_num_seeds=None):

        self.sim_type_path = sim_type_path	
        self.simulation_types = list(sim_type_path.keys())
        
        self.norm_type = norm_type   ## 0 : Use Orginal Data   1 : Z-Score Normalization   2 : [0 .. 1] Scaling			
        self.norm_label = InfoAnalysis.NORM_LABELS[norm_type]

        self.random_seed = random_seed
        self.rs = RandomState(random_seed)
        self.num_cores = num_cores

        assert (num_seeds_boostrapping==None) == (bootstrapping_runs==None), \
            "num_seeds_boostrapping and bootstrapping_runs should be both None or both not None"
        
        self.num_seeds_boostrapping = num_seeds_boostrapping # bootstrapping
        self.bootstrapping_runs = bootstrapping_runs # number of bootstrapping runs
        self.bootstrapping = num_seeds_boostrapping is not None # whether we are doing boostrapping

        self.restrict_to_first_n_converged_seeds = restrict_to_first_n_converged_seeds # whether to use only first n converged seed for analysis

        self.output_dir =output_dir # where to save plots, etc...
        if output_dir is not None:
            matplotlib.use('pdf')
            if not os.path.exists(self.output_dir):
                os.makedirs(output_dir)
            else:
                assert os.path.isdir(output_dir), \
                    f'Specified outputdir {output_dir} must be a dir not a file'

        self.debug = debug
        self.plot = plot

        # set to low number to test few seeds (not only converged), 
        # set to None to compute all seeds 
        # (or fewer if restrict_to_first_n_converged_seeds is not None)
        self.test_num_seeds = test_num_seeds 

        # self.lillieforsPValue = 0.05
        self.num_sim_types = len(self.simulation_types)
        self.num_sim_pairs = self.num_sim_types * (self.num_sim_types-1) /2
        self.BonferroniCorrection = float(0.05 / self.num_sim_types) ## divided by num settings 

        jp_kraskov_pkg = jp.JPackage("infodynamics.measures.continuous.kraskov")
        # http://lizier.me/joseph/software/jidt/javadocs/v1.5/

        self.condMultiMICalc = jp_kraskov_pkg.ConditionalMutualInfoCalculatorMultiVariateKraskov1()
        self.condMultiMICalc.setProperty("NOISE_LEVEL_TO_ADD", "0") # no noise for reproducibility

        self.multiVarMICalc = jp_kraskov_pkg.MutualInfoCalculatorMultiVariateKraskov1()
        self.multiVarMICalc.setProperty("NOISE_LEVEL_TO_ADD", "0") # no noise for reproducibility
        
        self.data = None # dictionary sim_type -> seed_dir -> sim_data		

    def checkDataNormality(self, M, whichData):
        [ksstats, pV] = lilliefors(M)
        print(whichData + ' KS-stats = ', ksstats, '  p-value = ', pV)											
        return pV

    # def performFriedman_n_PosthocWilcoxonTest(self, M, whichData, ylabel):
    # 	np.random.seed(self.random_seed) # reproducibility
    # 	print('\n====================================',  whichData, '\n')
    # 	self.plotBoxPlotList(M, self.simulation_types, whichData, ylabel)
    # 	# sys.exit()
    # 	# print(M[:, 0].shape, M[:, 1].shape, M[:, 2].shape)
    # 	[s, p] = friedmanchisquare(M[:, 0], M[:, 1], M[:, 2])			
    # 	print('Friedman Test -  ', whichData, ':  stat = ', s, '  p = ', p)
    # 	if p < self.BonferroniCorrection:				
    # 		for i in range(2):
    # 			for j in range(i + 1, 3, 1):
    # 				[sW, pW] = ranksums(M[:, i], M[:, j])
    # 				effectSize = abs(sW/np.sqrt(M.shape[0]))
    # 				print(self.simulation_types[i], ' vs. ', self.simulation_types[j], '  s = ', sW, '  p = ', pW, '  effect-size = ', effectSize, '(', \
    # 					info_utils.interpret_observed_effect_size(effectSize, 2), ')')
    # 				info_utils.show_descriptive_stats(M[:, i], self.simulation_types[i])
    # 				info_utils.show_descriptive_stats(M[:, j], self.simulation_types[j])


    def performKruskalWallis_n_PosthocWilcoxonTest(self, M, whichData):
        np.random.seed(self.random_seed) # reproducibility
        if self.debug:
            print('\n====================================',  whichData, '\n')
        
        h, p, etaSquaredEffectSize, epsilonSquaredEffectSize = None, None, None, None
        
        if M.shape[1]>2:
            [h, p] = kruskal(M[:, 0], M[:, 1], M[:, 2])
            etaSquaredEffectSize = (h - M.shape[1] + 1)/((M.shape[0] * M.shape[1]) - M.shape[1])
            epsilonSquaredEffectSize = h/(((M.shape[0] * M.shape[1])**2 - 1)/((M.shape[0] * M.shape[1]) + 1))

            if self.debug:
                print('Kruskal-Wallis Test -  ', whichData, ':  H-statistic = ', h, '  p = ', p, '  eta^2 = ', etaSquaredEffectSize, '(', \
                    info_utils.interpret_observed_effect_size(etaSquaredEffectSize, 1), '),  Epsilon^2 = ', epsilonSquaredEffectSize, ' (', \
                    info_utils.interpret_observed_effect_size(etaSquaredEffectSize, 1), ')')

        post_hoc_computation = M.shape[1]<=2 or self.bootstrapping or p < self.BonferroniCorrection

        post_hoc_stats = None
        if post_hoc_computation:
            all_pairs = list(combinations(list(range(self.num_sim_types)), 2)) # all pairs between sim types (e.g, [(0, 1), (0, 2), (1, 2)] for 3 sim_types)
            post_hoc_stats = np.zeros((len(all_pairs),3)) # for each pair among sim_types (row) we have 3 stats: sW, pW, effectSize
            for p_index, (i,j) in enumerate(all_pairs):
                [sW, pW] = ranksums(M[:, i], M[:, j])
                effectSize = abs(sW/np.sqrt(M.shape[0]))
                post_hoc_stats[p_index] = [sW, pW, effectSize]
                if self.debug:
                    print(self.simulation_types[i], ' vs. ', self.simulation_types[j], '  s = ', sW, '  p = ', pW, '  effect-size = ', effectSize, '(', \
                        info_utils.interpret_observed_effect_size(effectSize, 2), ')')
                    info_utils.show_descriptive_stats(M[:, i], self.simulation_types[i])
                    info_utils.show_descriptive_stats(M[:, j], self.simulation_types[j])					

        return h, p, etaSquaredEffectSize, epsilonSquaredEffectSize, post_hoc_stats

    def computeSpearmanCorr(self, M, distance, whichScenario, ylabel):
        np.random.seed(self.random_seed) # reproducibility
        fig = plt.figure(figsize = (40, 13))
        if self.norm_type != 0:
            if self.norm_type == 1:
                # TODO: fix this (meanDistGroup not defined)
                meanDistGroup = [(val - np.mean(meanDistGroup))/np.std(meanDistGroup) for val in meanDistGroup]
            else:
                # TODO: fix this (meanDistGroup not defined)
                meanDistGroup = [(val - min(meanDistGroup))/(max(meanDistGroup) - min(meanDistGroup)) for val in meanDistGroup]			
        whichMeasure = ['Conditional Mutual Information', 'Mutual Information', 'Co-Information']
        print('=======================  ', whichScenario, '  =======================')
        for i in range(M.shape[1]):
            ax1 = plt.subplot2grid((3, 1), (i, 0))

            b, m = polyfit(M[:, i], distance, 1)
            ax1.plot(M[:, i], distance, 'ro')
            ax1.plot(M[:, i], b + m * M[:, i], 'k-')
            # ax1.set_title(whichScenario + ' : ' + whichMeasure[i])				
            ax1.set_xlabel(whichScenario + ' : ' + whichMeasure[i], fontsize = 15)
            ax1.set_ylabel(ylabel, fontsize = 15)
            plt.xticks(fontsize = 15)
            plt.yticks(fontsize = 15)				

            [r, p] = spearmanr(M[:, i], distance)				
            if p < 0.05 and p < self.BonferroniCorrection:
                print(whichMeasure[i], ' vs. Target-Tracker-Distance: r = ', r, '  p-value = ', p, ' (Significant)')
            else:
                if p < 0.05 and p >= self.BonferroniCorrection:
                    print(whichMeasure[i], ' vs. Target-Tracker-Distance: r = ', r, '  p-value = ', p, ' (Non-significant After Bonferroni Correction)')
                else:
                    print(whichMeasure[i], ' vs. Target-Tracker-Distance: r = ', r, '  p-value = ', p, ' (Non-significant)')
        
        if self.output_dir is not None:
            output_file = os.path.join(self.output_dir, f'computeSpearmanCorr_{whichScenario}.pdf')
            plt.savefig(output_file)
        else:
            plt.show()
        

    def plotBoxPlotList(self, data, labels, ttle, ylabel):
        np.random.seed(self.random_seed) # reproducibility		
        plt.figure(figsize = (40, 13))
        sb.boxplot(data = data, showmeans = True,
            meanprops={"marker" : "o",
            "markerfacecolor" : "white", 
            "markeredgecolor" : "black",
            "markersize" : "10"})
        sb.stripplot(color='black', data = data)
        # plt.boxplot(a)
        # x = []
        # plot(x, a, 'r.', alpha=0.2)
        plt.xticks(range(0, len(labels)), labels, rotation = 0)
        plt.xticks(fontsize = 30)
        plt.yticks(fontsize = 25)
        plt.title(ttle)
        plt.ylabel(ylabel, fontsize = 25)

        if self.output_dir is not None:
            output_file = os.path.join(self.output_dir, f'plotBoxPlotList_{ttle}.pdf')
            plt.savefig(output_file)
        else:
            plt.show()

    def generateHeatMap(self, data, labels, ttle):
        fig = plt.figure(figsize = (40, 13))
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
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.title(ttle)

        if self.output_dir is not None:
            output_file = os.path.join(self.output_dir, f'generateHeatMap_{ttle}.pdf')
            plt.savefig(output_file)
        else:
            plt.show()


    def build_data(self):
        from pytictoc import TicToc

        t = TicToc() #create instance of class

        t.tic() #Start timer
        
        self.data = {} # dictionary sim_type -> seed_dir -> sim_data
        
        for sim_type, sim_dir in self.sim_type_path.items():
            
            print('Processing ', sim_type)

            self.data[sim_type] = sim_type_data = {}

            seeds = sorted([d for d in os.listdir(sim_dir) if d.startswith('seed_')])

            if self.test_num_seeds is not None:
                seeds = seeds[:self.test_num_seeds]

            if self.num_cores == 1:
                # single core
                for seed_dir in tqdm(seeds):				
                    dir = os.path.join(self.sim_type_path[sim_type], seed_dir)
                    _, seed_sim_data, converged = InfoAnalysis.get_simulation_results(dir)
                    if converged:
                        sim_type_data[seed_dir] = seed_sim_data
                        if len(sim_type_data) == self.restrict_to_first_n_converged_seeds:
                            break							
            else:
                # parallelization
                # seeds results is a list of tuples (seed_dir, sim_data) one per each seed
                seeds_results = Parallel(n_jobs=self.num_cores)(
                    delayed(InfoAnalysis.get_simulation_results)(os.path.join(self.sim_type_path[sim_type],dir)) \
                    for dir in tqdm(seeds)
                )				
                for seed_dir, seed_sim_data, converged in seeds_results:
                    if converged:
                        sim_type_data[seed_dir] = seed_sim_data
                        if len(sim_type_data) == self.restrict_to_first_n_converged_seeds:
                            break

            if self.restrict_to_first_n_converged_seeds is not None:
                assert( len(sim_type_data) == self.restrict_to_first_n_converged_seeds), \
                    f"Number of converged seeds ({len(sim_type_data)}) is less then required ({self.restrict_to_first_n_converged_seeds})"

            print(f'\tConverged: {len(sim_type_data)}')
                
        self.init_data_info()
        t.toc('Building data took') #Time elapsed since t.tic()

    @staticmethod
    def get_simulation_results(dir):
        perf, sim_perfs, evo, sim, data_record_list, sim_idx = run_simulation_from_dir(dir, quiet=True)
        seed_dir = os.path.basename(dir)
    
        simIndex = sim_perfs.index(min(sim_perfs))	  ### sim_perf is normalized, therefore, using 'minimum distance'
        # print(f'======  @ {seed_dir}', '   Sim', simIndex)
        
        sim_data = data_record_list[simIndex]
        converged = perf < analyze_results.CONVERGENCE_THRESHOLD
        return seed_dir, sim_data, converged

    
    def init_data_info(self):
        first_seed_sim_data = next(iter(self.data.values()))
        first_sim_data = next(iter(first_seed_sim_data.values()))
        self.num_trials, self.num_data_points = first_sim_data['delta_tracker_target'].shape


    def save_data_to_pickle(self, pickle_file):
        with open(pickle_file, 'wb') as handle:
                pickle.dump(self.data, handle, protocol = pickle.HIGHEST_PROTOCOL)


    def load_data_from_pickle(self, pickle_file):
        with open(pickle_file, 'rb') as handle:
            self.data = pickle.load(handle)		
        self.init_data_info()						
    
    def plot_seed_choices(self, sim_type_seed_counter):
        f,a = plt.subplots(1,3)
        for idx,ax in enumerate(a):
            sim_type = self.simulation_types[idx]
            sim_type_seeds_counter = sim_type_seed_counter[sim_type]
            seeds_values = list(sim_type_seeds_counter.keys())
            x_values = list(range(len(seeds_values)))
            y_values = list(sim_type_seeds_counter.values())
            ax.bar(x_values, y_values)
            ax.set_title(sim_type)
            # ax.set_xticks(list(range(len(seeds_values))), seeds_values)
        plt.tight_layout()

        if self.output_dir is not None:
            output_file = os.path.join(self.output_dir, f'plot_seed_choices.pdf')
            plt.savefig(output_file)
        else:
            plt.show()


        
def build_info_analysis_from_experiments(raw_args=None):

    import argparse
    from dol import data_path_utils

    # infodynamics.start_JVM()

    parser = argparse.ArgumentParser(
        description='InfoAnalysis'
    )

    parser.add_argument('--run_type', 
        type=str, 
        choices=[
            'overlapping_all_100_converged_no_bootstrapping', 
            'exc_switch_bootstrapping_12_seeds', 
            'exc_switch_first_100_converged',
            'alife_first_41_converged'		
        ], 
        required=True, 
        help='Types of run, choose one of the predefined strings'
    )
    parser.add_argument('--num_cores', type=int, default=1, help='Number of cores to used (defaults to 1)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output dir where to save plots (defaults to None: disply plots to screen)')
    parser.add_argument('--load_pickle', type=str, default=None, help='Specify pickle file where to load the data from')
    parser.add_argument('--save_pickle', type=str, default=None, help='Specify pickle file where to load the data from')

    args = parser.parse_args(raw_args)

    if args.run_type == 'overlapping_all_100_converged_no_bootstrapping':
        IA = InfoAnalysis(
            sim_type_path = data_path_utils.overlap_dir_xN(2), # overlap 3 neurons
            norm_type = 0,   ## 0 : Use Orginal Data   1 : Z-Score Normalization   2 : [0 .. 1] Scaling	
            num_cores = args.num_cores,
            random_seed = 1, # random seed used to initialize np.random.seed (for result reproducibility)		
            num_seeds_boostrapping = None, # specified min num of seeds to be used for bootstrapping (seed selection with replacement) - None (default) if no bootstrapping takes place (all sim type have same number of converged seeds)
            bootstrapping_runs = None, # number of boostrapping runs (default 100)
            restrict_to_first_n_converged_seeds = None, # whether to use only first n converged seed for analysis
            output_dir = args.output_dir,
            debug = True,
            plot = True,
            test_num_seeds = 5 # 5 # set to low number to test few seeds (not only converged), set to None to compute all seeds (or fewer if restrict_to_first_n_converged_seeds is not None)
        )
    elif args.run_type == 'exc_switch_bootstrapping_12_seeds':
        IA = InfoAnalysis(
            sim_type_path = data_path_utils.exc_switch_xN_dir(3), # exclusive + switch 3 neurons
            norm_type = 0,   ## 0 : Use Orginal Data   1 : Z-Score Normalization   2 : [0 .. 1] Scaling	
            num_cores = args.num_cores,
            random_seed = 1, # random seed used to initialize np.random.seed (for result reproducibility)		
            num_seeds_boostrapping = 12, # specified min num of seeds to be used for bootstrapping (seed selection with replacement) - None (default) if no bootstrapping takes place (all sim type have same number of converged seeds)
            bootstrapping_runs = 5000, # number of boostrapping runs (default 100)
            restrict_to_first_n_converged_seeds = None, # whether to use only first n converged seed for analysis
            output_dir = args.output_dir,
            debug = False,
            plot = True,
            test_num_seeds = None # 5 # set to low number to test few seeds (not only converged), set to None to compute all seeds (or fewer if restrict_to_first_n_converged_seeds is not None)
        )
    elif args.run_type == 'exc_switch_first_100_converged':
        IA = InfoAnalysis(
            sim_type_path = data_path_utils.exc_switch_xN_dir(3), # exclusive + switch 3 neurons
            norm_type = 0,   ## 0 : Use Orginal Data   1 : Z-Score Normalization   2 : [0 .. 1] Scaling	
            num_cores = args.num_cores,
            random_seed = 1, # random seed used to initialize np.random.seed (for result reproducibility)		
            num_seeds_boostrapping = None, # specified min num of seeds to be used for bootstrapping (seed selection with replacement) - None (default) if no bootstrapping takes place (all sim type have same number of converged seeds)
            bootstrapping_runs = None, # number of boostrapping runs (default 100)
            restrict_to_first_n_converged_seeds = 100, # whether to use only first n converged seed for analysis
            output_dir = args.output_dir,
            debug = True,
            plot = True,
            test_num_seeds = None # 5 # set to low number to test few seeds (not only converged), set to None to compute all seeds (or fewer if restrict_to_first_n_converged_seeds is not None)
        )		
    elif args.run_type == 'alife_first_41_converged':
        IA = InfoAnalysis(
            sim_type_path = data_path_utils.alife_dir_xN(2), # alife dir
            norm_type = 0,   ## 0 : Use Orginal Data   1 : Z-Score Normalization   2 : [0 .. 1] Scaling	
            num_cores = args.num_cores,
            random_seed = 1, # random seed used to initialize np.random.seed (for result reproducibility)		
            num_seeds_boostrapping = None, # specified min num of seeds to be used for bootstrapping (seed selection with replacement) - None (default) if no bootstrapping takes place (all sim type have same number of converged seeds)
            bootstrapping_runs = None, # number of boostrapping runs (default 100)
            restrict_to_first_n_converged_seeds = 41, # whether to use only first n converged seed for analysis
            output_dir = args.output_dir,
            debug = True,
            plot = True,
            test_num_seeds = None # 5 # set to low number to test few seeds (not only converged), set to None to compute all seeds (or fewer if restrict_to_first_n_converged_seeds is not None)
        )		
    else:
        assert False, 'Wron run_type type'
    
    # load/build/save data
    if args.load_pickle and os.path.exists(args.load_pickle):
        IA.load_data_from_pickle(args.load_pickle)
    else:
        IA.build_data()
    if args.save_pickle:
        IA.save_data_to_pickle(args.save_pickle)
    
    return IA


if __name__ == "__main__":
    build_info_analysis_from_experiments()

	
