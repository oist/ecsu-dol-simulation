#! /usr/bin/env python 3

from typing import Counter
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
from scipy.spatial.distance import pdist, squareform
from dol.run_from_dir import run_simulation_from_dir
from numpy.polynomial.polynomial import polyfit
from dol.run_from_dir import run_simulation_from_dir
from numpy.random import RandomState
from itertools import combinations

class InfoAnalysis:

    NORM_LABELS = {
        0: '',
        1: 'Z-Scored',
        2: '[0 ..1] Scaled'
    }	

    def __init__(self, agent_nodes, sim_type_path, whichNormalization, 
        random_seed, num_cores=1, num_seeds_boostrapping=None, bootstrapping_runs=None,
        restrict_to_first_n_converged_seeds = None,
        output_dir=None, debug=True, plot=True, test_num_seeds=None):

        self.initiJVM()

        self.agent_nodes = agent_nodes
        self.sim_type_path = sim_type_path	
        self.simulation_types = list(sim_type_path.keys())
        
        self.whichNormalization = whichNormalization   ## 0 : Use Orginal Data   1 : Z-Score Normalization   2 : [0 .. 1] Scaling			
        self.norm_label = InfoAnalysis.NORM_LABELS[whichNormalization]

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

    def initiJVM(self):
            jarLocation = os.path.join(os.getcwd(), "./", "infodynamics.jar")

            if (not(os.path.isfile(jarLocation))):
                exit("infodynamics.jar not found (expected at " + os.path.abspath(jarLocation) + ") - are you running from demos/python?")			
            jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation, convertStrings = False)   # convertStrings = False to silence the Warning while starting JVM 						

    def shutdownJVM(self):
        jp.shutdownJVM()

    def computeConditionalMultiVariateMutualInfo(self, agent1, agent2, target):
        if agent1.size == 0 or agent2.size == 0 or target.size == 0:
            print('Agent(s) or Traget Data Empty!')
            sys.exit()			
        
        self.condMultiMICalc.initialise(agent1.shape[1], agent2.shape[1], 1)		

        self.condMultiMICalc.setObservations(jp.JArray(jp.JDouble, 2)(agent1), jp.JArray(jp.JDouble, 2)(agent2), jp.JArray(jp.JDouble, 2)(target))
        result = self.condMultiMICalc.computeAverageLocalOfObservations()

        return result

    def computeMultiVariateMutualInfo(self, agent1, agent2):
        
        assert agent1.size != 0 and agent2.size != 0, 'One or Both Agent(s) Data Empty!'

        self.multiVarMICalc.initialise(agent1.shape[1], agent2.shape[1])		

        self.multiVarMICalc.setObservations(jp.JArray(jp.JDouble, 2)(agent1), jp.JArray(jp.JDouble, 2)(agent2))
        result = self.multiVarMICalc.computeAverageLocalOfObservations()

        return result

    def normalizeData(self, M):
        if self.whichNormalization == 0:
            return M
        scaler = preprocessing.StandardScaler().fit(M) if self.whichNormalization == 1 else preprocessing.MinMaxScaler().fit(M)
        return scaler.transform(M)

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
    # 					self.interpretObservedEffectSize(effectSize, 2), ')')
    # 				self.showDescriptiveStatistics(M[:, i], self.simulation_types[i])
    # 				self.showDescriptiveStatistics(M[:, j], self.simulation_types[j])

    def interpretObservedEffectSize(self, effectSize, whichOne):
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

    def performKruskalWallis_n_PosthocWilcoxonTest(self, M, whichData):
        np.random.seed(self.random_seed) # reproducibility
        if self.debug:
            print('\n====================================',  whichData, '\n')
        ylabel = f'{self.norm_label} {whichData}'
        if self.plot and not self.bootstrapping:
            self.plotBoxPlotList(M, self.simulation_types, whichData, ylabel)
        # sys.exit()
        # print(M[:, 0].shape, M[:, 1].shape, M[:, 2].shape)
        [h, p] = kruskal(M[:, 0], M[:, 1], M[:, 2])
        etaSquaredEffectSize = (h - M.shape[1] + 1)/((M.shape[0] * M.shape[1]) - M.shape[1])
        epsilonSquaredEffectSize = h/(((M.shape[0] * M.shape[1])**2 - 1)/((M.shape[0] * M.shape[1]) + 1))

        if self.debug:
            print('Kruskal-Wallis Test -  ', whichData, ':  H-statistic = ', h, '  p = ', p, '  eta^2 = ', etaSquaredEffectSize, '(', \
                self.interpretObservedEffectSize(etaSquaredEffectSize, 1), '),  Epsilon^2 = ', epsilonSquaredEffectSize, ' (', \
                self.interpretObservedEffectSize(etaSquaredEffectSize, 1), ')')

        post_hoc_computation = self.bootstrapping or p < self.BonferroniCorrection

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
                        self.interpretObservedEffectSize(effectSize, 2), ')')
                    self.showDescriptiveStatistics(M[:, i], self.simulation_types[i])
                    self.showDescriptiveStatistics(M[:, j], self.simulation_types[j])					

        return h, p, etaSquaredEffectSize, epsilonSquaredEffectSize, post_hoc_stats

    def showDescriptiveStatistics(self, data, whichOne):
        print('M-' + whichOne, ' = ', np.mean(data), ' SD-' + whichOne, ' = ', np.std(data), '  Mdn-' + whichOne, ' = ', np.median(data), \
            '  CI_95%-' + whichOne + ' = ', [np.percentile(data, 2.5), np.percentile(data, 97.5)])

    def computeSpearmanCorr(self, M, distance, whichScenario, ylabel):
        np.random.seed(self.random_seed) # reproducibility
        fig = plt.figure(figsize = (40, 13))
        if self.whichNormalization != 0:
            if self.whichNormalization == 1:
                meanDistGroup = [(val - np.mean(meanDistGroup))/np.std(meanDistGroup) for val in meanDistGroup]
            else:
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

        plt.savefig()


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


    def computeDistanceMetricsForSpecificSeed(self, whichSetting, whichSeed, trial_idx, whichDistance):
        if not whichSeed in list(set(os.listdir(self.sim_type_path[whichSetting]))):
            print(whichSeed, '  Is Not a Valid Seed')				
            sys.exit()

        data = self.data[whichSetting][whichSeed]

        agent1 = np.concatenate([data[node][trial_idx,0,:,:] for node in self.agent_nodes], axis=1)
        agent2 = np.concatenate([data[node][trial_idx,1,:,:] for node in self.agent_nodes], axis=1)
        agentsM = np.concatenate((agent1, agent2), axis = 1).T

        agentsM = self.normalizeData(agentsM)			

        agentsM = squareform(pdist(agentsM, whichDistance))

        labels = []
        cnt = 0
        for i in range(agentsM.shape[0]):
            if i < 6:
                labels.append('Node1_' + str(cnt + 1))
            else:
                if i == 6:
                    cnt = 0
                labels.append('Node2_' + str(cnt + 1))
            cnt += 1

        self.generateHeatMap(
            agentsM, 
            labels, 
            f'{whichSetting} {whichSeed} Trial {trial_idx+1} {whichDistance} Distance'
        )

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


    def compute_synergy(self):

        results = {} 
        # overall results sim_type -> sim_type_result
        # where sym_type_result is a dictionary: measure -> data (num_seed)
        nodes_measures = ['condMultVarMI', 'multVarMI', 'coinformation']
        dist_measures = ['trackerTargetDistMean', 'trackerTargetDistStd']
        results_measures = nodes_measures + dist_measures

        sim_type_num_seeds = {
            sim_type: len(seed_sim_data) # how many converged seeds in each sim type
            for sim_type, seed_sim_data in self.data.items()
        }

        all_sim_same_converged_seeds = len(set(sim_type_num_seeds.values())) == 1 # True
        min_num_converged_seeds = min(sim_type_num_seeds.values())

        assert self.bootstrapping or all_sim_same_converged_seeds, \
            f"Cannot compute statistics without bootstrapping if sim_type have different number of converged seeds: {sim_type_num_seeds}"
        
        assert not self.bootstrapping or self.num_seeds_boostrapping < min_num_converged_seeds, \
            f"You specified a num_seed_stats value that is >= to min number of converged seeds: {min_num_converged_seeds}"

        for sim_type, seed_sim_data in self.data.items():

            num_seeds = len(seed_sim_data)  # number of converged seeds

            sim_type_results = {
                measure: np.zeros((num_seeds, self.num_trials)) # num_seeds x num_trials (e.g., 100x4) eventually will turn into a num_seeds array (mean/std across columns)
                for measure in results_measures
            }

            for s, sim_data in enumerate(seed_sim_data.values()):

                # s is the seed index
                # sim_data is the dictionary with the results from the simulation

                delta_tracker_target = sim_data['delta_tracker_target'] # (num_trials, num_data_points)

                for t in range(self.num_trials):
                    # print('Trial # ', (t + 1))
                    agent1 = np.concatenate([sim_data[node][t,0,:,:] for node in self.agent_nodes], axis=1)
                    agent2 = np.concatenate([sim_data[node][t,1,:,:] for node in self.agent_nodes], axis=1)
                    target_pos = sim_data['target_position'][t]
                    sim_type_results['condMultVarMI'][s,t] = condMultVarMI = self.computeConditionalMultiVariateMutualInfo(agent1, agent2, np.expand_dims(target_pos, axis = 0).T)
                    sim_type_results['multVarMI'][s,t] = multVarMI = self.computeMultiVariateMutualInfo(agent1, agent2)
                    sim_type_results['coinformation'][s,t] = condMultVarMI - multVarMI  #### a.k.a interaction information, net synergy, and integration														
                    sim_type_results['trackerTargetDistMean'][s,t] = delta_tracker_target[t].mean()
                    sim_type_results['trackerTargetDistStd'][s,t] = delta_tracker_target[t].std()
                    
            # compute mean across trials
            # all variables will be 1-dim array with num_seeds elements
            for measure in sim_type_results:
                if measure == 'trackerTargetDistStd':
                    # we take the std across trials for std
                    sim_type_results[measure] = sim_type_results[measure].std(axis=1)	
                else:
                    # we take the mean across trials for all other values
                    sim_type_results[measure] = sim_type_results[measure].mean(axis=1) # 

            results[sim_type] = sim_type_results

        info_measures = {
            'condMultVarMI': 'Multivariate Conditional Mutual Information',				
            'multVarMI': 'Multivariate Mutual Information',
            'coinformation': 'Net-Synergy'
        }

        if self.plot:
            for measure, label in info_measures.items():
                results_measure_sim_types = [results[sim_type][measure] for sim_type in self.simulation_types]
                self.plotBoxPlotList(results_measure_sim_types, self.simulation_types, label, label)                        


        if self.bootstrapping:			
            # condMultVarMI = [results[sim_type]['condMultVarMI'] for sim_type in self.simulation_types] # num_sim_type rows x converged_seeds_in_sim_type
            # multVarMI = [results[sim_type]['multVarMI'] for sim_type in self.simulation_types]
            # coinformation = [results[sim_type]['coinformation'] for sim_type in self.simulation_types]			
            # TODO: boostrapping (random sampling with replacement)

            stats_factory = lambda: {
                'h': np.zeros(self.bootstrapping_runs),
                'p': np.zeros(self.bootstrapping_runs),
                'eta': np.zeros(self.bootstrapping_runs),
                'epsilon': np.zeros(self.bootstrapping_runs),
                'post_hoc_stats': np.zeros((self.bootstrapping_runs,self.num_sim_types, 3))
            }

            boostrapping_stats = {
                measure: stats_factory()
                for measure in info_measures
            }
            
            sim_type_seed_idx_counter = {
                sim_type: Counter()
                for sim_type in self.simulation_types
            }

            for b in tqdm(range(self.bootstrapping_runs)):	

                for measure, label in info_measures.items():					

                    selected_stat_indexes = np.zeros((self.num_sim_types, self.num_seeds_boostrapping), dtype="int")
                    selected_stat = np.zeros((self.num_sim_types, self.num_seeds_boostrapping))
                    
                    for i, sim_type in enumerate(self.simulation_types):
                        indexes = list(range(len(results[sim_type][measure])))
                        selected_stat_indexes[i] = self.rs.choice(indexes, size=self.num_seeds_boostrapping, replace=True)
                        selected_stat[i] = np.take(results[sim_type][measure], selected_stat_indexes[i])

                    for sim_index, sim_type_seed_idx_choices in enumerate(selected_stat_indexes):
                        sim_type = self.simulation_types[sim_index]
                        sim_type_seed_idx_counter[sim_type].update(sim_type_seed_idx_choices)

                    selected_stat = self.normalizeData(selected_stat) 
                
                    ################# We might want to check whether data follows normal distribution and if positive apply parametric tests instead.

                    # self.checkDataNormality(selected_stat.flatten().tolist(), label)

                    h, p, eta, epsilon, post_hoc_stats = self.performKruskalWallis_n_PosthocWilcoxonTest(selected_stat, label)

                    boostrapping_stats_measure = boostrapping_stats[measure]
                    boostrapping_stats_measure['h'][b] = h
                    boostrapping_stats_measure['p'][b] = p
                    boostrapping_stats_measure['eta'][b] = eta
                    boostrapping_stats_measure['epsilon'][b] = epsilon
                    boostrapping_stats_measure['post_hoc_stats'][b] = post_hoc_stats

            if self.plot:
                self.plot_seed_choices(sim_type_seed_idx_counter)

            for measure, label in info_measures.items():
                # TODO: maybe only print those stats for which mean(p) < bonferroni ...
                boostrapping_stats_measure = boostrapping_stats[measure]
                print(label)
                for sub_measure, data in boostrapping_stats_measure.items():					
                    self.showDescriptiveStatistics(data, sub_measure)

                
        else:
            # no bootstrpping
            # following arrays have shape num_seeds x num_sim_types (e.g., 100 x 3)
            condMultVarMI = np.array([results[sim_type]['condMultVarMI'] for sim_type in self.simulation_types]).T
            multVarMI = np.array([results[sim_type]['multVarMI'] for sim_type in self.simulation_types]).T
            coinformation = np.array([results[sim_type]['coinformation'] for sim_type in self.simulation_types]).T
            
            condMultVarMI = self.normalizeData(condMultVarMI) 
            multVarMI = self.normalizeData(multVarMI)
            coinformation = self.normalizeData(coinformation)

            ################# We might want to check whether data follows normal distribution and if positive apply parametric tests instead.

            # self.checkDataNormality(condMultVarMI.flatten().tolist(), 'Multivariate Conditional Mutual Information')
            # self.checkDataNormality(multVarMI.flatten().tolist(), 'Multivariate Mutual Information')
            # self.checkDataNormality(coinformation.flatten().tolist(), 'Net-Synergy')			

            self.performKruskalWallis_n_PosthocWilcoxonTest(condMultVarMI, f'Multivariate Conditional Mutual Information')
            self.performKruskalWallis_n_PosthocWilcoxonTest(multVarMI, f'Multivariate Mutual Information')
            self.performKruskalWallis_n_PosthocWilcoxonTest(coinformation, f'Net-Synergy')		

            print('\n\n Spearman Correlation Based on Target-Tracker Mean Distance')

            for sim_type, sim_type_results in results.items():
                cond_mult_coinfo_mean = np.array([sim_type_results[m] for m in nodes_measures]).T
                self.computeSpearmanCorr(
                    cond_mult_coinfo_mean, 
                    sim_type_results['trackerTargetDistMean'], 
                    sim_type + '_Mean', 
                    'Mean Target-Tracker Disatnce'
                )  ##### 1 : z-scored   2 : [0 .. 1] scaled

            print('\n\n Spearman Correlation Based on Target-Tracker SD Distance')

            for sim_type, sim_type_results in results.items():
                cond_mult_coinfo_mean = np.array([sim_type_results[m] for m in nodes_measures]).T
                self.computeSpearmanCorr(
                    cond_mult_coinfo_mean, 
                    sim_type_results['trackerTargetDistStd'], 
                    sim_type + '_SD', 
                    'SD Target-Tracker Disatnce'
                )		

        

if __name__ == "__main__":
    import argparse
    from dol import data_path_utils

    parser = argparse.ArgumentParser(
        description='Synergy'
    )

    parser.add_argument('--run_type', 
        type=str, 
        choices=[
            'overlapping_all_100_converged_no_bootstrapping', 
            'exc_switch_bootstrapping_12_seeds', 
            'exc_switch_first_100_converged'			
        ], 
        required=True, 
        help='Types of run, choose one of the predefined strings'
    )
    parser.add_argument('--cores', type=int, default=1, help='Number of cores to used (defaults to 1)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output dir where to save plots (defaults to None: disply plots to screen)')

    args = parser.parse_args()


    agent_nodes = ['agents_brain_input', 'agents_brain_state', 'agents_brain_output']
    
    load_data = False # set to True if data is read from pickle (has to be saved beforehand)
    save_data = False # set to True if data will be saved to pickle (to be loaded faster successively)
    
    if args.run_type == 'overlapping_all_100_converged_no_bootstrapping':
        IA = InfoAnalysis(
            agent_nodes = agent_nodes, 
            sim_type_path = data_path_utils.overlap_dir_xN(2), # overlap 2 neurons
            whichNormalization = 0,   ## 0 : Use Orginal Data   1 : Z-Score Normalization   2 : [0 .. 1] Scaling	
            num_cores = args.cores,
            random_seed = 1, # random seed used to initialize np.random.seed (for result reproducibility)		
            num_seeds_boostrapping = None, # specified min num of seeds to be used for bootstrapping (seed selection with replacement) - None (default) if no bootstrapping takes place (all sim type have same number of converged seeds)
            bootstrapping_runs = None, # number of boostrapping runs (default 100)
            restrict_to_first_n_converged_seeds = None, # whether to use only first n converged seed for analysis
            output_dir = args.output_dir,
            debug = True,
            plot = True,
            test_num_seeds = None # 5 # set to low number to test few seeds (not only converged), set to None to compute all seeds (or fewer if restrict_to_first_n_converged_seeds is not None)
        )
    elif args.run_type == 'exc_switch_bootstrapping_12_seeds':
        IA = InfoAnalysis(
            agent_nodes = agent_nodes, 
            sim_type_path = data_path_utils.exc_switch_xN_dir(3), # exclusive + switch 3 neurons
            whichNormalization = 0,   ## 0 : Use Orginal Data   1 : Z-Score Normalization   2 : [0 .. 1] Scaling	
            num_cores = args.cores,
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
            agent_nodes = agent_nodes, 
            sim_type_path = data_path_utils.exc_switch_xN_dir(3), # exclusive + switch 3 neurons
            whichNormalization = 0,   ## 0 : Use Orginal Data   1 : Z-Score Normalization   2 : [0 .. 1] Scaling	
            num_cores = args.cores,
            random_seed = 1, # random seed used to initialize np.random.seed (for result reproducibility)		
            num_seeds_boostrapping = None, # specified min num of seeds to be used for bootstrapping (seed selection with replacement) - None (default) if no bootstrapping takes place (all sim type have same number of converged seeds)
            bootstrapping_runs = None, # number of boostrapping runs (default 100)
            restrict_to_first_n_converged_seeds = 100, # whether to use only first n converged seed for analysis
            output_dir = args.output_dir,
            debug = True,
            plot = True,
            test_num_seeds = None # 5 # set to low number to test few seeds (not only converged), set to None to compute all seeds (or fewer if restrict_to_first_n_converged_seeds is not None)
        )		
    else:
        assert False, 'Wron run_type type'
    
    # load/build/save data
    pickle_path = 'results/synergy.pickle' # where data is saved/loaded
    if load_data and os.path.exists(pickle_path):
        IA.load_data_from_pickle(pickle_path)
    else:
        IA.build_data()
    if save_data:
        IA.save_data_to_pickle(pickle_path)
    
    # main computation script
    IA.compute_synergy()

    ''' 
    correlation = 1 - corr(x, y)  AND  canberra = \sum_i (abs(x_i - y_i))/(abs(x_i) + abs(y_i))
    '''
    # distanceMetrics = ['cosine', 'correlation', 'euclidean', 'cityblock', 'canberra']   
    # distanceMetrics = ['correlation']   
    # for metric in distanceMetrics:
    # 	IA.computeDistanceMetricsForSpecificSeed('individual', 'seed_001', 0, metric)

    IA.shutdownJVM()			
