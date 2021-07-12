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
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import friedmanchisquare, ranksums, kruskal, spearmanr, pearsonr
import scipy.special as special
from scipy.spatial.distance import pdist, squareform
from dol.run_from_dir import run_simulation_from_dir
from numpy.polynomial.polynomial import polyfit
from dol.run_from_dir import run_simulation_from_dir

class InfoAnalysis:

	def __init__(self, agent_nodes, sim_type_path, whichNormalization, max_num_seeds=None):
		self.initiJVM()

		self.agent_nodes = agent_nodes
		self.sim_type_path = sim_type_path	
		self.simulation_types = sim_type_path.keys()
		self.whichNormalization = whichNormalization   ## 0 : Use Orginal Data   1 : Z-Score Normalization   2 : [0 .. 1] Scaling			
		self.max_num_seeds = max_num_seeds # restrict it to first n seeds (for test purpose)

		# self.lillieforsPValue = 0.05
		self.BonferroniCorrection = float(0.05 / len(self.simulation_types)) ## divided by num settings 

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

		condMultiMICalcClass = jp.JPackage("infodynamics.measures.continuous.kraskov").ConditionalMutualInfoCalculatorMultiVariateKraskov1
		condMultiMICalc = condMultiMICalcClass()
		condMultiMICalc.initialise(agent1.shape[1], agent2.shape[1], 1)		

		condMultiMICalc.setObservations(jp.JArray(jp.JDouble, 2)(agent1), jp.JArray(jp.JDouble, 2)(agent2), jp.JArray(jp.JDouble, 2)(target))
		result = condMultiMICalc.computeAverageLocalOfObservations()

		return result

	def computeMultiVariateMutualInfo(self, agent1, agent2):
		
		assert agent1.size != 0 and agent2.size != 0, 'One or Both Agent(s) Data Empty!'

		multiVarMIClass = jp.JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
		multiVarMICalc = multiVarMIClass()

		multiVarMICalc.initialise(agent1.shape[1], agent2.shape[1])		

		multiVarMICalc.setObservations(jp.JArray(jp.JDouble, 2)(agent1), jp.JArray(jp.JDouble, 2)(agent2))
		result = multiVarMICalc.computeAverageLocalOfObservations()

		return result

	def compute_seed_trial_means(self, info_results):		
		simData = []
		meanDist = []
		stdDist = []

		# num_seeds = len(info_results)

		for seed_dir, seed_info_results in info_results.items():
			seed_data = []
			seed_tracker_mean = []
			seed_tracker_std = []
			for seed_info_results_trial in seed_info_results:
				(condMultVarMI, multVarMI, coinformation, trackerTargetDist) = seed_info_results_trial
				seed_data.append([condMultVarMI, multVarMI, coinformation])
				seed_tracker_mean.append(trackerTargetDist.mean())
				seed_tracker_std.append(trackerTargetDist.std())
			
			seed_means = np.array(seed_data).mean(axis = 0).tolist() # average across trials of (condMultVarMI, multVarMI, coinformation)
			meanDist.append(np.mean(seed_tracker_mean)) # average across trials of seed_tracker_mean
			stdDist.append(np.mean(seed_tracker_std)) # average across trials of seed_tracker_std
			
			simData.append(seed_means)
		
		simData = np.array(simData)

		assert len(simData) == len(meanDist) == len(stdDist)

		print(simData, '   ', simData.shape, '   ', len(meanDist), '  ', len(stdDist))
		
		return simData, meanDist, stdDist


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

	def performKruskalWallis_n_PosthocWilcoxonTest(self, M, whichData, ylabel):
		print('\n====================================',  whichData, '\n')
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

	def computeSpearmanCorr(self, M, distance, whichScenario, whichScaling, yLabel):
		fig = plt.figure(figsize = (10, 6))
		if whichScaling != 0:
			if whichScaling == 1:
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
			ax1.set_xlabel(whichScenario + ' : ' + whichMeasure[i])
			ax1.set_ylabel(yLabel)
			# plt.xticks(fontsize = 15)
			# plt.yticks(fontsize = 15)				

			[r, p] = spearmanr(M[:, i], distance)				
			if p < 0.05 and p < self.BonferroniCorrection:
				print(whichMeasure[i], ' vs. Target-Tracker-Distance: r = ', r, '  p-value = ', p, ' (Significant)')
			else:
				if p < 0.05 and p >= self.BonferroniCorrection:
					print(whichMeasure[i], ' vs. Target-Tracker-Distance: r = ', r, '  p-value = ', p, ' (Non-significant After Bonferroni Correction)')
				else:
					print(whichMeasure[i], ' vs. Target-Tracker-Distance: r = ', r, '  p-value = ', p, ' (Non-significant)')
		plt.show()

	def plotBoxPlotList(self, data, labels, ttle, yLabel):
		np.random.seed(1) # same seed_dir to have same jitter (distribution of points along x axis)
		plt.figure(figsize = (10, 6))
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
		# plt.xticks(fontsize = 30)
		# plt.yticks(fontsize = 25)
		plt.title(ttle)
		plt.ylabel(yLabel)
		plt.show()						


	def generateHeatMap(self, data, labels, ttle):
		fig = plt.figure(figsize = (10, 6))
		ax = fig.add_subplot(111)
		cax = ax.matshow(data, cmap = cm.Spectral_r, interpolation = 'nearest')
		fig.colorbar(cax)

		xaxis = np.arange(len(labels))
		ax.set_xticks(xaxis)
		ax.set_yticks(xaxis)
		ax.set_xticklabels(labels, rotation = 90)
		ax.xaxis.set_ticks_position('bottom')
		ax.set_yticklabels(labels)
		# plt.xticks(fontsize = 15)
		# plt.yticks(fontsize = 15)
		plt.title(ttle)

		plt.show()				

	def computeDistanceMetricsForSpecificSeed(self, whichSetting, whichSeed, trial_idx, normalizationFlag, whichDistance):
		if not whichSeed in list(set(os.listdir(self.sim_type_path[whichSetting]))):
			print(whichSeed, '  Is Not a Valid Seed')				
			sys.exit()

		dir = os.path.join(self.sim_type_path[whichSetting], whichSeed)
		perf, sim_perfs, evo, sim, data_record_list, sim_idx = run_simulation_from_dir(dir)			

		simIndex = sim_perfs.index(min(sim_perfs))	  ### sim_perf is normalized, therefore, using 'minimum distance'			
		
		data = data_record_list[simIndex]

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

	def read_data(self):
		
		self.data = {} # dictionary sim_type -> seed_dir -> sim_data
		
		for sim_type, sim_dir in self.sim_type_path.items(): 				
			
			print('Processing ', sim_type)

			self.data[sim_type] = {}

			seeds = sorted([d for d in os.listdir(sim_dir) if d.startswith('seed_')])

			if self.max_num_seeds is not None:
				seeds = seeds[:self.max_num_seeds]

			for seed_dir in seeds:
				
				dir = os.path.join(self.sim_type_path[sim_type], seed_dir)

				perf, sim_perfs, evo, sim, data_record_list, sim_idx = run_simulation_from_dir(dir)
	
				simIndex = sim_perfs.index(min(sim_perfs))	  ### sim_perf is normalized, therefore, using 'minimum distance'
				print(f'======  @ {seed_dir}', '   Sim', simIndex)
				
				sim_data = data_record_list[simIndex]

				self.data[sim_type][seed_dir] = sim_data


	def save_data_to_pickle(self, pickle_file):
		with open(pickle_file, 'wb') as handle:
				pickle.dump(self.data, handle, protocol = pickle.HIGHEST_PROTOCOL)


	def load_data_from_pickle(self, pickle_file):
		with open(pickle_file, 'rb') as handle:
			self.data = pickle.load(handle)								


	def compute_synergy(self):

		results = {}
		
		for sim_type, seed_sim_data in self.data.items():

			info_results = {} 

			for seed_dir, sim_data in seed_sim_data.items():
				
				seed_info_results = info_results[seed_dir] = []

				num_trials = len(sim_data['trials_performances'])
				
				for trialIndex in range(num_trials):
					# print('Trial # ', (trialIndex + 1))
					# for each trial we save the following tuple: (condMultVarMI, multVarMI, coinformation, trackerTargetDist)

					agent1 = np.concatenate([sim_data[node][trialIndex,0,:,:] for node in self.agent_nodes], axis=1)
					agent2 = np.concatenate([sim_data[node][trialIndex,1,:,:] for node in self.agent_nodes], axis=1)
					target_pos = sim_data['target_position'][trialIndex]
					condMultVarMI = self.computeConditionalMultiVariateMutualInfo(agent1, agent2, np.expand_dims(target_pos, axis = 0).T)
					multVarMI = self.computeMultiVariateMutualInfo(agent1, agent2)
					coinformation = condMultVarMI - multVarMI  #### a.k.a interaction information, net synergy, and integration			
					trackerTargetDist = sim_data['delta_tracker_target'][trialIndex]												

					seed_info_results.append(
						(condMultVarMI, multVarMI, coinformation, trackerTargetDist)
					)

			results[sim_type] = self.compute_seed_trial_means(info_results) # cond_mult_coinfo_mean, meanDist, stdDist


		condMultVarMI = np.array([results[sim_type][0][:,0] for sim_type in self.simulation_types]).T
		multVarMI = np.array([results[sim_type][0][:,1] for sim_type in self.simulation_types]).T
		coinformation = np.array([results[sim_type][0][:,2] for sim_type in self.simulation_types]).T

		condMultVarMI = self.normalizeData(condMultVarMI) 
		multVarMI = self.normalizeData(multVarMI)
		coinformation = self.normalizeData(coinformation)


		if self.whichNormalization == 0:
			yLabel = ''			
		elif self.whichNormalization == 1:
			yLabel = 'Z-Scored '
		elif self.whichNormalization == 2:
			yLabel = '[0 ..1] Scaled '

		################# We might want to check whether data follows normal distribution and if positive apply parametric tests instead.

		# self.checkDataNormality(condMultVarMI.flatten().tolist(), 'Multivariate Conditional Mutual Information')
		# self.checkDataNormality(multVarMI.flatten().tolist(), 'Multivariate Mutual Information')
		# self.checkDataNormality(coinformation.flatten().tolist(), 'Net-Synergy')			

		self.performKruskalWallis_n_PosthocWilcoxonTest(condMultVarMI, 'Multivariate Conditional Mutual Information', yLabel + 'Multivariate Conditional Mutual Information')
		self.performKruskalWallis_n_PosthocWilcoxonTest(multVarMI, 'Multivariate Mutual Information', yLabel + 'Multivariate Mutual Information')
		self.performKruskalWallis_n_PosthocWilcoxonTest(coinformation, 'Net-Synergy', yLabel + 'Net-Synergy')		

		print('\n\n Spearman Correlation Based on Target-Tracker Mean Distance')

		for sim_type in self.simulation_types:
			cond_mult_coinfo_mean, meanDist, stdDist = results[sim_type]
			self.computeSpearmanCorr(
				cond_mult_coinfo_mean, 
				meanDist, 
				sim_type, 
				self.whichNormalization, 
				'Mean Target-Tracker Disatnce'
			)  ##### 1 : z-scored   2 : [0 .. 1] scaled

		print('\n\n Spearman Correlation Based on Target-Tracker SD Distance')

		for sim_type in self.simulation_types:
			cond_mult_coinfo_mean, meanDist, stdDist = results[sim_type]
			self.computeSpearmanCorr(
				cond_mult_coinfo_mean, 
				stdDist, 
				sim_type, 
				self.whichNormalization, 
				'SD Target-Tracker Disatnce'
			)		

		

if __name__ == "__main__":
	#############  At present and due to the small and unbalanced number of seeds in Switch Setting, the code is primarily meant and tested 
	############# on Overlap Setting. Given the zero # of converged seeds in the case individual in Switch Setting, the code will not proceed
	############# to analysis.
	from dol.data_path_utils import overlap_dir_xN, exc_switch_xN_dir
	agent_nodes = ['agents_brain_input', 'agents_brain_state', 'agents_brain_output']
	overlap_data_dirs = overlap_dir_xN(2) # overlap 2 neurons
	exc_switch_data_dirs = exc_switch_xN_dir(2) # exclusive + swtich 2 neurons
	
	IA = InfoAnalysis(
		agent_nodes=agent_nodes, 
		sim_type_path=overlap_data_dirs,
		whichNormalization = 0,   ## 0 : Use Orginal Data   1 : Z-Score Normalization   2 : [0 .. 1] Scaling			
		max_num_seeds=5
	)
	
	IA.read_data()
	IA.save_data_to_pickle('results/synergy.pickle')
	# IA.load_data_from_pickle('results/synergy.pickle')
	IA.compute_synergy()

	# ''' 
	# correlation = 1 - corr(x, y)  AND  canberra = \sum_i (abs(x_i - y_i))/(abs(x_i) + abs(y_i))
	# '''
	# distanceMetrics = ['cosine', 'correlation', 'euclidean', 'cityblock', 'canberra']   
	# for metric in distanceMetrics:
	# 	IA.computeDistanceMetricsForSpecificSeed('individual', 'seed_010', 1, 0, metric)

	IA.shutdownJVM()						