#! /usr/bin/env python 3


import sys
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sb

import jpype as jp

from sklearn import preprocessing

from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import friedmanchisquare, ranksums, f_oneway

class infoAnalysis:
	def __init__(self):
		try:
			self.initiJVM()

			self.generation = 5000
			self.acceptedSeeds = [2, 3, 6, 10, 11, 12, 13, 14, 15, 16, 18, 19]  ## This is the list of valid seeds
			self.dataFolders = ['1d_2n_exc-0.1_zfill_rp-0_switch', '1d_2n_exc-0.1_zfill_rp-3_np-4_switch', '1d_2n_exc-0.1_zfill_rp-3_switch']
			self.lillieforsPValue = 0.05
			self.BonferroniCorrection = float(0.05 / 3) ## divided by three since we have three settings 
			self.whichNormalization = 2   ## 0 : Use Orginal Data   1 : Z-Score Normalization   2 : [0 .. 1] Scaling

			self.includedNodes = ['agents_brain_input', 'agents_brain_state', 'agents_brain_output', 'target_position']

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
			agent1 = np.concatenate((data[selectedKeys[0]][trialIndex][0], np.concatenate((data[selectedKeys[1]][trialIndex][0], data[selectedKeys[2]][trialIndex][0]), axis = 1)), axis = 1)
			agent2 = np.concatenate((data[selectedKeys[0]][trialIndex][1], np.concatenate((data[selectedKeys[1]][trialIndex][1], data[selectedKeys[2]][trialIndex][1]), axis = 1)), axis = 1)
			target = data[selectedKeys[3]][trialIndex]
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
			condMultVarMI = []
			multVarMI = []
			coinformation = []
			for seed in self.acceptedSeeds:
				with open(addr + f'seed_{str(seed).zfill(3)}' + '/resultsDic.pickle', 'rb') as handle:
					data = pickle.load(handle)						
					handle.close()	
					print('====   ', f'seed_{str(seed).zfill(3)}')
					# print(data)
					simData = []
					for j in range(4):
						tmp.append([data[list(data.keys())[0]]['trial' + str(j + 1)]['condMultVarMI'], data[list(data.keys())[0]]['trial' + str(j + 1)]['multVarMI'], \
							data[list(data.keys())[0]]['trial' + str(j + 1)]['coinformation']])
					print(np.array(tmp))
					simData.append(np.array(tmp).mean(axis = 0).tolist())
					np.array(tmp).mean(axis = 0).tolist()

					sys.exit()
					# for j in range(4):
					# 	print(data.keys)


					for i in range(3):
						tmp = []
						# print('Sim' + str(i + 1))
						for j in range(4):
							tmp.append([data['sim' + str(i + 1)]['trial' + str(j + 1)]['condMultVarMI'], data['sim' + str(i + 1)]['trial' + str(j + 1)]['multVarMI'], \
								data['sim' + str(i + 1)]['trial' + str(j + 1)]['coinformation']])
						simData.append(np.array(tmp).mean(axis = 0).tolist())					
					condMultVarMI.append(np.array(simData)[:, 0].tolist())
					multVarMI.append(np.array(simData)[:, 1].tolist())
					coinformation.append(np.array(simData)[:, 2].tolist())
			
			print('Multivariate Conditional Mututal Information: ', np.array(condMultVarMI).shape)
			print('Multivariate Mututal Information: ', np.array(multVarMI).shape)
			print('Net-Synergy: ', np.array(coinformation).shape)

			return np.array(condMultVarMI), np.array(multVarMI), np.array(coinformation)
		except Exception as e:
			print('@ prepareDataForAnalysis() :  ', e)
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
			self.plotBoxPlotList(M, ['Sim1', 'Sim2', 'Sim3'], whichData, ylabel)
			# sys.exit()
			print(M[:, 0].shape, M[:, 1].shape, M[:, 2].shape)
			[s, p] = friedmanchisquare(M[:, 0], M[:, 1], M[:, 2])			
			print('Friedman Test -  ', whichData, ':  stat = ', s, '  p = ', p)
			if p < self.BonferroniCorrection:				
				for i in range(2):
					for j in range(i + 1, 3, 1):
						[sW, pW] = ranksums(M[i], M[j])
						print('Sim ' + str(i + 1) + ' vs. Sim' + str(j + 1), '  s = ', sW, '  p = ', pW, '  effect-size = ', sW/np.sqrt(M.shape[0]))
						self.showDescriptiveStatistics(M[i], i + 1)
						self.showDescriptiveStatistics(M[j], j + 1)
		except Exception as e:
			print('performFriedman_n_PosthocWilcoxonTest() :  ', e)
			sys.exit()

	def showDescriptiveStatistics(self, data, whichOne):
		try:
			print('M' + str(whichOne), ' = ', np.mean(data), ' SD' + str(whichOne), ' = ', np.std(data), '  Mdn' + str(whichOne), ' = ', np.median(data), \
				'  CI_95%_' + str(whichOne) + ' = ', [np.percentile(data, 2.5), np.percentile(data, 97.5)])
		except Exception as e:
			print('@ showDescriptiveStatistics() :  ', e)
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
			plt.xticks(fontsize = 15)
			plt.yticks(fontsize = 15)
			plt.title(ttle)
			plt.ylabel(yLabel, fontsize = 20)
			plt.show()						
		except Exception as e:
			print('plotBoxPlotList() :  ', e)
			sys.exit()			

