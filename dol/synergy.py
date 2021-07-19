#! /usr/bin/env python 3

import sys
import os
import numpy as np
import pickle
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

class InfoAnalysis:

	NORM_LABELS = {
		0: '',
		1: 'Z-Scored',
		2: '[0 ..1] Scaled'
	}	

	def __init__(self, agent_nodes, sim_type_path, whichNormalization, random_seed, num_cores=1, max_num_seeds=None):
		self.initiJVM()

		self.agent_nodes = agent_nodes
		self.sim_type_path = sim_type_path	
		self.simulation_types = list(sim_type_path.keys())
		
		self.whichNormalization = whichNormalization   ## 0 : Use Orginal Data   1 : Z-Score Normalization   2 : [0 .. 1] Scaling			
		self.norm_label = InfoAnalysis.NORM_LABELS[whichNormalization]

		self.random_seed = random_seed
		self.num_cores = num_cores

		self.max_num_seeds = max_num_seeds # restrict it to first n seeds (for test purpose)

		# self.lillieforsPValue = 0.05
		self.BonferroniCorrection = float(0.05 / len(self.simulation_types)) ## divided by num settings 

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

	def performFriedman_n_PosthocWilcoxonTest(self, M, whichData, ylabel):
		np.random.seed(self.random_seed) # reproducibility
		print('\n====================================',  whichData, '\n')
		self.plotBoxPlotList(M, self.simulation_types, whichData, ylabel)
		# sys.exit()
		# print(M[:, 0].shape, M[:, 1].shape, M[:, 2].shape)
		[s, p] = friedmanchisquare(M[:, 0], M[:, 1], M[:, 2])			
		print('Friedman Test -  ', whichData, ':  stat = ', s, '  p = ', p)
		if p < self.BonferroniCorrection:				
			for i in range(2):
				for j in range(i + 1, 3, 1):
					[sW, pW] = ranksums(M[:, i], M[:, j])
					effectSize = abs(sW/np.sqrt(M.shape[0]))
					print(self.simulation_types[i], ' vs. ', self.simulation_types[j], '  s = ', sW, '  p = ', pW, '  effect-size = ', effectSize, '(', \
						self.interpretObservedEffectSize(effectSize, 2), ')')
					self.showDescriptiveStatistics(M[:, i], self.simulation_types[i])
					self.showDescriptiveStatistics(M[:, j], self.simulation_types[j])

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
		print('\n====================================',  whichData, '\n')
		ylabel = f'{self.norm_label} {whichData}'
		self.plotBoxPlotList(M, self.simulation_types, whichData, ylabel)
		# sys.exit()
		# print(M[:, 0].shape, M[:, 1].shape, M[:, 2].shape)
		[h, p] = kruskal(M[:, 0], M[:, 1], M[:, 2])
		etaSquaredEffectSize = (h - M.shape[1] + 1)/((M.shape[0] * M.shape[1]) - M.shape[1])
		epsilonSquaredEffectSize = h/(((M.shape[0] * M.shape[1])**2 - 1)/((M.shape[0] * M.shape[1]) + 1))

		print('Kruskal-Wallis Test -  ', whichData, ':  H-statistic = ', h, '  p = ', p, '  eta^2 = ', etaSquaredEffectSize, '(', \
			self.interpretObservedEffectSize(etaSquaredEffectSize, 1), '),  Epsilon^2 = ', epsilonSquaredEffectSize, ' (', \
			self.interpretObservedEffectSize(etaSquaredEffectSize, 1), ')')
		if p < self.BonferroniCorrection:				
			for i in range(2):
				for j in range(i + 1, 3, 1):
					[sW, pW] = ranksums(M[:, i], M[:, j])
					effectSize = abs(sW/np.sqrt(M.shape[0]))
					print(self.simulation_types[i], ' vs. ', self.simulation_types[j], '  s = ', sW, '  p = ', pW, '  effect-size = ', effectSize, '(', \
						self.interpretObservedEffectSize(effectSize, 2), ')')
					self.showDescriptiveStatistics(M[:, i], self.simulation_types[i])
					self.showDescriptiveStatistics(M[:, j], self.simulation_types[j])

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
		plt.show()						


	def generateHeatMap(self, data, labels, ttle):
		fig = plt.figure(figsize = (40, 13))
		ax = fig.add_subplot(111)
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

			if self.max_num_seeds is not None:
				seeds = seeds[:self.max_num_seeds]

			if self.num_cores == 1:
				# single core
				for seed_dir in seeds:				
					dir = os.path.join(self.sim_type_path[sim_type], seed_dir)
					_, seed_sim_data = InfoAnalysis.get_simulation_results(dir)
					sim_type_data[seed_dir] = seed_sim_data
			else:
				# parallelization
				# seeds results is a list of tuples (seed_dir, sim_data) one per each seed
				seeds_results = Parallel(n_jobs=self.num_cores)(
                    delayed(InfoAnalysis.get_simulation_results)(os.path.join(self.sim_type_path[sim_type],dir)) \
                    for dir in seeds
                )
				for seed_dir, seed_sim_data in seeds_results:
					sim_type_data[seed_dir] = seed_sim_data
				
		self.init_data_info()
		t.toc('Building data took') #Time elapsed since t.tic()

	@staticmethod
	def get_simulation_results(dir):
		perf, sim_perfs, evo, sim, data_record_list, sim_idx = run_simulation_from_dir(dir, quiet=True)
		seed_dir = os.path.basename(dir)
	
		simIndex = sim_perfs.index(min(sim_perfs))	  ### sim_perf is normalized, therefore, using 'minimum distance'
		# print(f'======  @ {seed_dir}', '   Sim', simIndex)
		
		sim_data = data_record_list[simIndex]
		return seed_dir, sim_data

	
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


	def compute_synergy(self):

		results = {} 
		# overall results sim_type -> sim_type_result
		# where sym_type_result is a dictionary: measure -> data (num_seed)
		nodes_measures = ['condMultVarMI', 'multVarMI', 'coinformation']
		dist_measures = ['trackerTargetDistMean', 'trackerTargetDistStd']
		results_measures = nodes_measures + dist_measures
		
		for sim_type, seed_sim_data in self.data.items():
			
			num_seeds = len(seed_sim_data)			

			sim_type_results = {
				measure: np.zeros((num_seeds, self.num_trials))
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
					sim_type_results[measure] = sim_type_results[measure].mean(axis=1)

			results[sim_type] = sim_type_results


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
				sim_type, 
				'Mean Target-Tracker Disatnce'
			)  ##### 1 : z-scored   2 : [0 .. 1] scaled

		print('\n\n Spearman Correlation Based on Target-Tracker SD Distance')

		for sim_type, sim_type_results in results.items():
			cond_mult_coinfo_mean = np.array([sim_type_results[m] for m in nodes_measures]).T
			self.computeSpearmanCorr(
				cond_mult_coinfo_mean, 
				sim_type_results['trackerTargetDistStd'], 
				sim_type, 
				'SD Target-Tracker Disatnce'
			)		

		

if __name__ == "__main__":
	#############  At present and due to the small and unbalanced number of seeds in Switch Setting, the code is primarily meant and tested 
	############# on Overlap Setting. Given the zero # of converged seeds in the case individual in Switch Setting, the code will not proceed
	############# to analysis.
	from dol.data_path_utils import overlap_dir_xN, exc_switch_xN_dir

	agent_nodes = ['agents_brain_input', 'agents_brain_state', 'agents_brain_output']
	
	# directory structures with experiments and seeds
	# exp_type -> dir_path
	# exp_types are ['individual', 'group', 'joint']
	overlap_data_dirs = overlap_dir_xN(2) # overlap 2 neurons
	exc_switch_data_dirs = exc_switch_xN_dir(2) # exclusive + swtich 2 neurons
	
	pickle_path = 'results/synergy.pickle' # where data is saved/loaded
	
	load_data = False # set to True if data is read from pickle (has to be saved beforehand)
	save_data = False # set to True if data will be saved to pickle (to be loaded faster successively)
	
	IA = InfoAnalysis(
		agent_nodes = agent_nodes, 
		sim_type_path = overlap_data_dirs,
		whichNormalization = 0,   ## 0 : Use Orginal Data   1 : Z-Score Normalization   2 : [0 .. 1] Scaling	
		num_cores = 1,
		random_seed = 1, # random seed used to initialize np.random.seed (for result reproducibility)		
		# max_num_seeds = 5 # set to low number to test few seeds, set to None to compute all seeds
	)
	
	if load_data and os.path.exists(pickle_path):
		IA.load_data_from_pickle(pickle_path)
	else:
		IA.build_data()
	
	if save_data:
		IA.save_data_to_pickle(pickle_path)
	
	IA.compute_synergy()

	''' 
	correlation = 1 - corr(x, y)  AND  canberra = \sum_i (abs(x_i - y_i))/(abs(x_i) + abs(y_i))
	'''
	# distanceMetrics = ['cosine', 'correlation', 'euclidean', 'cityblock', 'canberra']   
	distanceMetrics = ['correlation']   
	for metric in distanceMetrics:
		IA.computeDistanceMetricsForSpecificSeed('individual', 'seed_001', 0, metric)

	IA.shutdownJVM()			
