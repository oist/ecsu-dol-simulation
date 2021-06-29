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

class infoAnalysis:
	def __init__(self, whichSimSetup):
		try:
			self.initiJVM()

			phil_trans_si_data = os.path.join('data', 'phil_trans_si')

			if whichSimSetup == 1:
				print('Switch Setting')
				self.dataFolders = {
				    'group': os.path.join( # 8 seeds converged: [1, 5, 8, 9, 12, 14, 15, 19]
				        phil_trans_si_data, 
				        '1d_2n_exc-0.1_zfill_rp-3_switch'
				    ),
				    'joint': os.path.join(
				        phil_trans_si_data, # 1 seed converged: [6]
				        '1d_2n_exc-0.1_zfill_rp-0_switch'
				    ),
				    'individual': os.path.join(
				        phil_trans_si_data, # 0 seeds converged: []
				        '1d_2n_exc-0.1_zfill_rp-3_np-4_switch'
				    )
				}
			elif whichSimSetup == 2:
				print('Overlap Setting')
				self.dataFolders = {
				    'group': os.path.join( # 20/20 seeds converged
				        phil_trans_si_data, 
				        '1d_2n_zfill_rp-3_overlap'
				    ),
				    'joint': os.path.join( # 20/20 seeds converged
				        phil_trans_si_data, 
				        '1d_2n_zfill_rp-0_overlap'
				    ),
				    'individual': os.path.join( # 20/20 seeds converged
				        phil_trans_si_data, 
				        '1d_2n_zfill_rp-3_np-4_overlap'
				    )
				}				

			self.generation = 5000
			# self.acceptedSeeds = [2, 3, 6, 10, 11, 12, 13, 14, 15, 16, 18, 19]  ## This is the list of valid seeds
			# self.dataFolders = ['1d_2n_exc-0.1_zfill_rp-0_switch', '1d_2n_exc-0.1_zfill_rp-3_np-4_switch', '1d_2n_exc-0.1_zfill_rp-3_switch']
			self.lillieforsPValue = 0.05
			self.BonferroniCorrection = float(0.05 / 3) ## divided by three since we have three settings 
			self.whichNormalization = 0   ## 0 : Use Orginal Data   1 : Z-Score Normalization   2 : [0 .. 1] Scaling

			# ['genotypes', 'phenotypes', 'delta_tracker_target', 'target_position', 'target_velocity', 'tracker_position', 'tracker_angle', 'tracker_wheels', 'tracker_velocity', 'tracker_signals', 'agents_motors_control_indexes', 'agents_sensors', \
			# 'agents_brain_input', 'agents_brain_state', 'agents_derivatives', 'agents_brain_output', 'agents_motors', 'info']

			self.includedNodes = ['agents_brain_input', 'agents_brain_state', 'agents_brain_output', 'target_position']
			# self.includedNodes = ['agents_sensors', 'agents_brain_output', 'target_position']
			self.xTicksLabel = ['Individual', 'Group', 'Joint']

			self.resultFolder = './results/MultVarMI_CondMi_CoInfo/'

		except Exception as e:
			print('@ infoAnalysis() init -- ', e)
			sys.exit()

	def initiJVM(self):
		try:
			jarLocation = os.path.join(os.getcwd(), "./", "infodynamics.jar")

			if (not(os.path.isfile(jarLocation))):
				exit("infodynamics.jar not found (expected at " + os.path.abspath(jarLocation) + ") - are you running from demos/python?")			
			jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation, convertStrings = False)   # convertStrings = False to silence the Warning while starting JVM 						
		except Exception as e:
			print('@ initiJVM() :  ', e)
			sys.exit()

	def checkIfResultsGenerated(self):
		try:
			if not os.path.exists(self.resultFolder):
				os.makedirs(self.resultFolder)
				return 0
			return 1
			# # print(list(os.listdir(self.resultFolder)))
			# tmp = list(os.listdir(self.resultFolder))
			# if '.DS_Store' in tmp:
			# 	tmp.remove('.DS_Store')
			# return len(tmp)
		except Exception as e:
			print('@ checkIfResultsGenerated() :  ', e)
			sys.exit()
	def shutdownJVM(self):
		try:
			jp.shutdownJVM()
		except Exception as e:
			print('@ shutdownJVM() :  ', e)
			sys.exit()

	def returnAgentsTargetData(self, data, selectedKeys, trialIndex):
		try:
			if len(selectedKeys) == 0:
				print('Keys Missing')
				sys.exit()
			# print(data.keys())		
			# sys.exit()
			# print(selectedKeys)
			agent1 = []
			agent2 = []
			for keyIndex in range(len(selectedKeys) - 1):
				if len(agent1) == 0:
					agent1 = data[selectedKeys[keyIndex]][trialIndex][0]
				else:
					agent1 = np.concatenate((agent1, data[selectedKeys[keyIndex]][trialIndex][0]), axis = 1)
				if len(agent2) == 0:
					agent2 = data[selectedKeys[keyIndex]][trialIndex][1]
				else:
					agent2 = np.concatenate((agent2, data[selectedKeys[keyIndex]][trialIndex][1]), axis = 1)					
			print(agent1.shape, '  ', agent2.shape)

			# agent1 = np.concatenate((data[selectedKeys[0]][trialIndex][0], np.concatenate((data[selectedKeys[1]][trialIndex][0], data[selectedKeys[2]][trialIndex][0]), axis = 1)), axis = 1)
			# agent2 = np.concatenate((data[selectedKeys[0]][trialIndex][1], np.concatenate((data[selectedKeys[1]][trialIndex][1], data[selectedKeys[2]][trialIndex][1]), axis = 1)), axis = 1)

			target = data[selectedKeys[len(selectedKeys) - 1]][trialIndex]
			# print(target.shape)			
			# sys.exit()
			return agent1, agent2, target
		except Exception as e:
			print('@ returnAgentsData() :  ', e)
			sys.exit()

	def computeConditionalMultiVariateMutualInfo(self, agent1, agent2, target):
		try:			
			if agent1.size == 0 or agent2.size == 0 or target.size == 0:
				print('Agent(s) or Traget Data Empty!')
				sys.exit()			

			condMultiMICalcClass = jp.JPackage("infodynamics.measures.continuous.kraskov").ConditionalMutualInfoCalculatorMultiVariateKraskov1
			condMultiMICalc = condMultiMICalcClass()

			# print(agent1.shape, '  ', type(agent1))
			# print(agent2.shape, '  ', type(agent2))
			# print(target.shape, '  ', type(target))			

			condMultiMICalc.initialise(agent1.shape[1], agent2.shape[1], 1)		

			condMultiMICalc.setObservations(jp.JArray(jp.JDouble, 2)(agent1), jp.JArray(jp.JDouble, 2)(agent2), jp.JArray(jp.JDouble, 2)(target))
			result = condMultiMICalc.computeAverageLocalOfObservations()
			# print('Conditional Multi-Variate MI: ', result)
			return result

			# self.shutdownJVM()

		except Exception as e:
			print('@ computeConditionalMutualInfo() :  ', e)
			sys.exit()

	def computeMultiVariateMutualInfo(self, agent1, agent2):
		try:
			if agent1.size == 0 or agent2.size == 0:
				print('One or Both Agent(s) Data Empty!')
				sys.exit()
			multiVarMIClass = jp.JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
			multiVarMICalc = multiVarMIClass()

			# print(agent1.shape, '  ', type(agent1))
			# print(agent2.shape, '  ', type(agent2))

			multiVarMICalc.initialise(agent1.shape[1], agent2.shape[1])		

			multiVarMICalc.setObservations(jp.JArray(jp.JDouble, 2)(agent1), jp.JArray(jp.JDouble, 2)(agent2))
			result = multiVarMICalc.computeAverageLocalOfObservations()
			# print("Multi-Variate MI = %.4f nats" % (result))
			return result

			# tmp = []
			# for i in range(agent2.shape[1]):
			# 	multiVarMICalc.initialise(6, 1)			
			# 	multiVarMICalc.setObservations(jp.JArray(jp.JDouble, 2)(agent1), jp.JArray(jp.JDouble, 2)(np.expand_dims(agent2[:, i], axis = 0).T))
			# 	result = multiVarMICalc.computeAverageLocalOfObservations()
			# 	print(str(i) + "Multi-Variate MI = %.4f nats" % (result))			
			# 	tmp.append(result)
			# print(np.mean(result))

		except Exception as e:
			print('@ computeMultiVariateMutualInfo() :  ', e)
			sys.exit()			

	def saveResults(self, addr, seedName, resultsDic):
		try:
			if len(addr) == 0:
				print('Destination Address is Missing!')
				sys.exit()
			if len(seedName) == 0:
				print('Seed Name is Missing!')
				sys.exit()				
			if not os.path.exists(addr):
				os.makedirs(addr)

			if not os.path.exists(addr + seedName):
				os.mkdir(addr + seedName)				

			with open(addr + seedName + '/resultsDic.pickle', 'wb') as handle:
				pickle.dump(resultsDic, handle, protocol = pickle.HIGHEST_PROTOCOL)
				handle.close()

		except Exception as e:
			print('@ saveResults() :  ', e)
			sys.exit()

	def prepareDataForAnalysis(self, addr):
		try:
			simData = []
			meanDist = []
			stdDist = []

			acceptedSeeds = list(os.listdir(addr))
			print('++++   ', addr)
			for seed in acceptedSeeds:
				with open(addr + '/' + seed + '/resultsDic.pickle', 'rb') as handle:
					data = pickle.load(handle)						
					handle.close()	
					print('====   ', f'seed_{str(seed).zfill(3)}')
					# print(data)
					# sys.exit()					
					tmp = []
					dM = []
					dSTD = []
					for j in range(4):
						tmp.append([data[list(data.keys())[0]]['trial' + str(j + 1)]['condMultVarMI'], data[list(data.keys())[0]]['trial' + str(j + 1)]['multVarMI'], \
							data[list(data.keys())[0]]['trial' + str(j + 1)]['coinformation']])
						dM.append(data[list(data.keys())[0]]['trial' + str(j + 1)]['trackerTargetDist'].mean())
						dSTD.append(data[list(data.keys())[0]]['trial' + str(j + 1)]['trackerTargetDist'].std())						
					# print(np.array(tmp))					
					simData.append(np.array(tmp).mean(axis = 0).tolist())
					meanDist.append(np.mean(dM))
					stdDist.append(np.std(dSTD))
			print(np.array(simData), '   ', np.array(simData).shape, '   ', len(meanDist), '  ', len(stdDist))

			return np.array(simData), meanDist, stdDist
		except Exception as e:
			print('@ prepareDataForAnalysis() :  ', e)
			sys.exit()

	def extract_n_CombineGivenMeasureValues(self, M1, M2, M3, measureIndex):
		try:
			return np.array([M1[:, measureIndex], M2[:, measureIndex], M3[:, measureIndex]]).T
		except Exception as e:
			print('@ extract_n_CombineGivenMeasureValues() :  ', e)
			sys.exit()

	def normalizeData(self, M, whichScaling):
		try:
			scaler = preprocessing.StandardScaler().fit(M) if whichScaling == 1 else preprocessing.MinMaxScaler().fit(M)

			return scaler.transform(M)
		except Exception as e:
			print('@ normalizeData() :  ', e)
			sys.exit()

	def checkDataNormality(self, M, whichData):
		try:
			[ksstats, pV] = lilliefors(M)
			print(whichData + ' KS-stats = ', ksstats, '  p-value = ', pV)											
			return pV
		except Exception as e:
			print('@ checkDataNormality() :  ', e)
			sys.exit()

	def performFriedman_n_PosthocWilcoxonTest(self, M, whichData, ylabel):
		try:
			print('\n====================================',  whichData, '\n')
			self.plotBoxPlotList(M, self.xTicksLabel, whichData, ylabel)
			# sys.exit()
			# print(M[:, 0].shape, M[:, 1].shape, M[:, 2].shape)
			[s, p] = friedmanchisquare(M[:, 0], M[:, 1], M[:, 2])			
			print('Friedman Test -  ', whichData, ':  stat = ', s, '  p = ', p)
			if p < self.BonferroniCorrection:				
				for i in range(2):
					for j in range(i + 1, 3, 1):
						[sW, pW] = ranksums(M[:, i], M[:, j])
						effectSize = abs(sW/np.sqrt(M.shape[0]))
						print(self.xTicksLabel[i], ' vs. ', self.xTicksLabel[j], '  s = ', sW, '  p = ', pW, '  effect-size = ', effectSize, '(', \
							self.interpretObservedEffectSize(effectSize, 2), ')')
						self.showDescriptiveStatistics(M[:, i], self.xTicksLabel[i])
						self.showDescriptiveStatistics(M[:, j], self.xTicksLabel[j])
		except Exception as e:
			print('performFriedman_n_PosthocWilcoxonTest() :  ', e)
			sys.exit()

	def interpretObservedEffectSize(self, effectSize, whichOne):
		try:
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

		except Exception as e:
			print('@ interpretObservedEffectSize() :  ', e)
			sys.exit()

	def performKruskalWallis_n_PosthocWilcoxonTest(self, M, whichData, ylabel):
		try:
			print('\n====================================',  whichData, '\n')
			self.plotBoxPlotList(M, self.xTicksLabel, whichData, ylabel)
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
						print(self.xTicksLabel[i], ' vs. ', self.xTicksLabel[j], '  s = ', sW, '  p = ', pW, '  effect-size = ', effectSize, '(', \
							self.interpretObservedEffectSize(effectSize, 2), ')')
						self.showDescriptiveStatistics(M[:, i], self.xTicksLabel[i])
						self.showDescriptiveStatistics(M[:, j], self.xTicksLabel[j])
		except Exception as e:
			print('performKruskalWallis_n_PosthocWilcoxonTest() :  ', e)
			sys.exit()	

	def showDescriptiveStatistics(self, data, whichOne):
		try:
			print('M-' + whichOne, ' = ', np.mean(data), ' SD-' + whichOne, ' = ', np.std(data), '  Mdn-' + whichOne, ' = ', np.median(data), \
				'  CI_95%-' + whichOne + ' = ', [np.percentile(data, 2.5), np.percentile(data, 97.5)])
		except Exception as e:
			print('@ showDescriptiveStatistics() :  ', e)
			sys.exit()

	def computeSpearmanCorr(self, M, distance, whichScenario, whichScaling):
		try:
			if whichScaling != 0:
				if whichScaling == 1:
					meanDistGroup = [(val - np.mean(meanDistGroup))/np.std(meanDistGroup) for val in meanDistGroup]
				else:
					meanDistGroup = [(val - min(meanDistGroup))/(max(meanDistGroup) - min(meanDistGroup)) for val in meanDistGroup]			
			whichMeasure = ['Conditional Mutual Information', 'Mutual Information', 'Co-Information']
			print('=======================  ', whichScenario, '  =======================')
			for i in range(M.shape[1]):
				[r, p] = spearmanr(M[:, i], distance)
				if p < 0.05 and p < self.BonferroniCorrection:
					print(whichMeasure[i], ' vs. Target-Tracker-Distance: r = ', r, '  p-value = ', p, ' (Significant)')
				else:
					if p < 0.05 and p >= self.BonferroniCorrection:
						print(whichMeasure[i], ' vs. Target-Tracker-Distance: r = ', r, '  p-value = ', p, ' (Non-significant After Bonferroni Correction)')
					else:
						print(whichMeasure[i], ' vs. Target-Tracker-Distance: r = ', r, '  p-value = ', p, ' (Non-significant)')
		except Exception as e:
			print('@ computeSpearmanCorr() :  ', e)
			sys.exit()

	def plotBoxPlotList(self, data, labels, ttle, yLabel):
		try:
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
			plt.ylabel(yLabel, fontsize = 25)
			plt.show()						
		except Exception as e:
			print('plotBoxPlotList() :  ', e)
			sys.exit()			

	def saveAgentsAveragedDataOverAllSeeds_n_Trials(self, A1, A2, whichScenario):
		try:
			# print(np.array(A1).shape, '  ', np.array(A2).shape)
			# print(np.array(A1).mean(axis = 0).shape, '  ', np.array(A2).mean(axis = 0).shape)
			# print(np.array(A1).mean(axis = 0))
			# print('++++++++++++++++++++++++++')
			# print(np.array(A2).mean(axis = 0))
			agentsData = {'A1' : A1, 'A2' : A2}

			with open(self.resultFolder[0 : self.resultFolder.index('Mult')] + whichScenario + '_AgentsAllSeedsTrialsAvgNodes.pickle', 'wb') as handle:
				pickle.dump(agentsData, handle, protocol = pickle.HIGHEST_PROTOCOL)
				handle.close()			
		except Exception as e:
			print('@ saveAgentsAveragedDataOverAllSeeds_n_Trials() :  ', e)
			sys.exit()

	def returnAgentsAverageDataFileNames(self):
		try:
			tmp = os.listdir(self.resultFolder[0 : self.resultFolder.index('Mult')])
			if '.DS_Store' in tmp:
				tmp.remove('.DS_Store')
			agentsFiles = []			
			for fName in tmp:
				if 'AgentsAllSeedsTrialsAvgNodes' in fName:
					agentsFiles.append(fName)
			return agentsFiles
		except Exception as e:
			print('@ returnAgentsAverageDataFileNames() :  ', e)
			sys.exit()

	def computeDistanceMetrics(self, whichDistance, normalizationFlag):
		try:
			agentsFiles = self.returnAgentsAverageDataFileNames()
			for fName in agentsFiles:
				print(fName)
				with open(self.resultFolder[0 : self.resultFolder.index('Mult')] + fName, 'rb') as handle:
					data = pickle.load(handle)
					handle.close()
					A1 = np.array(data['A1']).mean(axis = 0)
					A2 = np.array(data['A2']).mean(axis = 0)
					print(A1.shape, '  ', A2.shape)
					agentsM = np.concatenate((A1, A2), axis = 1).T
					print(agentsM.shape)			

					agentsM = squareform(pdist(agentsM, whichDistance))

					if normalizationFlag != 0:
						agentsM = self.normalizeData(agentsM, normalizationFlag)

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

					self.generateHeatMap(agentsM, labels, fName[0 : fName.index('_')] + '  -  ' + whichDistance + ' Distance')								

		except Exception as e:
			print('@ computeDistanceMetrics() :  ', e)
			sys.exit()


	def generateHeatMap(self, data, labels, ttle):
		try:			
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
		except Exception as e:
			print('@ generateHeatMap() :  ', e)
			sys.exit()			