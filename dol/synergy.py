#! /usr/bin/env python 3

import sys
import numpy as np
import os
from dol.run_from_dir import run_simulation_from_dir

from dol.synergyAnalysisClass import infoAnalysis
import sys


if __name__ == "__main__":
	try:
		#############  At present and due to the small and unbalanced number of seeds in Switch Setting, the code is primarily meant and tested 
		############# on Overlap Setting. Given the zero # of converged seeds in the case individual in Switch Setting, the code will not proceed
		############# to analysis.
		whichSetting = 2  #### 1 : Switch Setting   2 : Overlap Setting
		Obj = infoAnalysis(whichSetting)
		# print(Obj.checkIfResultsGenerated())
		# sys.exit()
		if Obj.checkIfResultsGenerated() == 0:
			simulationSettings = list(Obj.dataFolders.keys())
			for settingIndex in range(len(simulationSettings)):
				print('Processing ', simulationSettings[settingIndex])

				A1 = []
				A2 = []

				seeds = list(set(os.listdir(Obj.dataFolders[simulationSettings[settingIndex]])))
				if '.DS_Store' in seeds:
					seeds.remove('.DS_Store')
				for seed in seeds:
					dir = Obj.dataFolders[simulationSettings[settingIndex]] + '/' + seed
					# print(dir)
					# sys.exit()
					perf, sim_perfs, evo, sim, data_record_list, sim_idx = run_simulation_from_dir(dir = dir, generation = Obj.generation)

					results = {}		
					simIndex = sim_perfs.index(min(sim_perfs))	  ### sim_perf is normalized, therefore, using 'minimum distance'
					print(f'======  @ seed_{str(seed).zfill(3)}', '   Sim', simIndex)
					if 'sim' + str(simIndex + 1) not in results:
						results['sim' + str(simIndex + 1)] = {}

					for trialIndex in range(len(data_record_list[simIndex]['agents_brain_output'])):
						print('Trial # ', (trialIndex + 1))

						agent1, agent2, target = Obj.returnAgentsTargetData(data_record_list[simIndex], Obj.includedNodes, trialIndex)			
						A1.append(agent1)
						A2.append(agent2)
						# print(agent1.shape, '  ', agent2.shape, '  ', target.shape)
						# sys.exit()
						condMultVarMI = Obj.computeConditionalMultiVariateMutualInfo(agent1, agent2, np.expand_dims(target, axis = 0).T)
						multVarMI = Obj.computeMultiVariateMutualInfo(agent1, agent2)

						results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)] = {}
						results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)]['condMultVarMI'] = condMultVarMI
						results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)]['multVarMI'] = multVarMI
						results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)]['coinformation'] = condMultVarMI - multVarMI  #### a.k.a interaction information, net synergy, and integration					
						results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)]['trackerTargetDist'] = data_record_list[simIndex]['delta_tracker_target'][trialIndex]												

					Obj.saveResults(Obj.resultFolder + Obj.dataFolders[simulationSettings[settingIndex]][Obj.dataFolders[simulationSettings[settingIndex]].find('_si/') + 4 : \
						len(Obj.dataFolders[simulationSettings[settingIndex]])] + '/', seed, results)

				Obj.saveAgentsAveragedDataOverAllSeeds_n_Trials(A1, A2, simulationSettings[settingIndex])

		simResultsDirs = list(os.listdir(Obj.resultFolder))				
		if '.DS_Store' in simResultsDirs:
			simResultsDirs.remove('.DS_Store')
		
		for resultDirIndex in range(len(simResultsDirs)):
			# print(simResultsDirs[resultDirIndex], '  ', Obj.resultFolder + simResultsDirs[resultDirIndex])
			if whichSetting == 2:
				if resultDirIndex == 0:
					individual, meanDistIndividual, stdDistIndividual = Obj.prepareDataForAnalysis(Obj.resultFolder + simResultsDirs[resultDirIndex])
				elif resultDirIndex == 1:
					group, meanDistGroup, stdDistGroup = Obj.prepareDataForAnalysis(Obj.resultFolder + simResultsDirs[resultDirIndex])
				elif resultDirIndex == 2:
					joint, meanDistJoint, stdDistJoint = Obj.prepareDataForAnalysis(Obj.resultFolder + simResultsDirs[resultDirIndex])

		# print(individual.shape, '  ', len(meanDistIndividual), '  ', len(stdDistIndividual))
		# print(group.shape, '  ', len(meanDistGroup), '   ', len(stdDistGroup))
		# print(joint.shape, '   ', len(meanDistJoint), '   ', len(stdDistJoint))

		condMultVarMI = Obj.extract_n_CombineGivenMeasureValues(individual, group, joint, 0)
		multVarMI = Obj.extract_n_CombineGivenMeasureValues(individual, group, joint, 1)
		coinformation = Obj.extract_n_CombineGivenMeasureValues(individual, group, joint, 2)		

		# print(np.array(condMultVarMI).shape)
		# print(np.array(multVarMI).shape)
		# print(np.array(coinformation).shape)

		if Obj.whichNormalization != 0 :
			condMultVarMI = Obj.normalizeData(condMultVarMI, Obj.whichNormalization) 
			multVarMI = Obj.normalizeData(multVarMI, Obj.whichNormalization)
			coinformation = Obj.normalizeData(coinformation, Obj.whichNormalization)

		# print(condMultVarMI)
		# print(multVarMI)
		# print(coinformation)

		# Obj.performFriedman_n_PosthocWilcoxonTest(condMultVarMI, 'Multivariate Conditional Mutual Information', 'Z-Scored Multivariate Conditional Mutual Information' \
		# 	if Obj.whichNormalization == 1 else '[0 ..1] Scaled Multivariate Conditional Mutual Information')
		# Obj.performFriedman_n_PosthocWilcoxonTest(multVarMI, 'Multivariate Mutual Information', 'Z-Scored Multivariate Mutual Information' \
		# 	if Obj.whichNormalization == 1 else '[0 ..1] Scaled Multivariate Mutual Information')
		# Obj.performFriedman_n_PosthocWilcoxonTest(coinformation, 'Net-Synergy', 'Z-Scored Net-Synergy' \
		# 	if Obj.whichNormalization == 1 else '[0 ..1] Scaled Multivariate Net-Synergy')

		if Obj.whichNormalization == 0:
			yLabel = ''			
		elif Obj.whichNormalization == 1:
			yLabel = 'Z-Scored '
		elif Obj.whichNormalization == 2:
			yLabel = '[0 ..1] Scaled '

		################# We might want to check whether data follows normal distribution and if positive apply parametric tests instead.

		# Obj.checkDataNormality(condMultVarMI.flatten().tolist(), 'Multivariate Conditional Mutual Information')
		# Obj.checkDataNormality(multVarMI.flatten().tolist(), 'Multivariate Mutual Information')
		# Obj.checkDataNormality(coinformation.flatten().tolist(), 'Net-Synergy')			

		Obj.performKruskalWallis_n_PosthocWilcoxonTest(condMultVarMI, 'Multivariate Conditional Mutual Information', yLabel + 'Multivariate Conditional Mutual Information')
		Obj.performKruskalWallis_n_PosthocWilcoxonTest(multVarMI, 'Multivariate Mutual Information', yLabel + 'Multivariate Mutual Information')
		Obj.performKruskalWallis_n_PosthocWilcoxonTest(coinformation, 'Net-Synergy', yLabel + 'Net-Synergy')		

		print('\n\n Spearman Correlation Based on Target-Tracker Mean Distance')

		Obj.computeSpearmanCorr(individual, meanDistIndividual, 'Individual - Mean Target-Tracker Disatnce', Obj.whichNormalization)  ##### 1 : z-scored   2 : [0 .. 1] scaled
		Obj.computeSpearmanCorr(group, meanDistGroup, 'Group - Mean Target-Tracker Disatnce', Obj.whichNormalization)		
		Obj.computeSpearmanCorr(joint, meanDistJoint, 'Joint - Mean Target-Tracker Disatnce', Obj.whichNormalization)		

		print('\n\n Spearman Correlation Based on Target-Tracker SD Distance')

		Obj.computeSpearmanCorr(individual, stdDistIndividual, 'Individual - SD Target-Tracker Disatnce', Obj.whichNormalization)		
		Obj.computeSpearmanCorr(group, stdDistGroup, 'Group - SD Target-Tracker Disatnce', Obj.whichNormalization)		
		Obj.computeSpearmanCorr(joint, stdDistJoint, 'Joint - SD Target-Tracker Disatnce', Obj.whichNormalization)						
		
		Obj.computeDistanceMetrics('cosine', 0)   ###  0 : use original data   1 : Z-score normalization     2 : Scale within [0 .. 1] interval

		# import pickle
		# import numpy as np
		# for fName in agentsFiles:
		# 	print(fName)
		# 	with open(Obj.resultFolder[0 : Obj.resultFolder.index('Mult')] + fName, 'rb') as handle:
		# 		data = pickle.load(handle)
		# 		handle.close()
		# 		A1 = np.array(data['A1']).mean(axis = 0)
		# 		A2 = np.array(data['A2']).mean(axis = 0)
		# 		print(A1.shape, '  ', A2.shape)
		# 		print(np.concatenate((A1, A2), axis = 1).T.shape)

		Obj.shutdownJVM()						
		
	except Exception as e:
		print('@ synrgy.py main() :  ', e)
		sys.exit()