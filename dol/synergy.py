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
		if Obj.checkIfResultsGenerated() == 0:
			simulationSettings = list(Obj.dataFolders.keys())
			for settingIndex in range(len(simulationSettings)):
				print('Processing ', simulationSettings[settingIndex])

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
						# print(agent1.shape, '  ', agent2.shape, '  ', target.shape)
						# sys.exit()
						condMultVarMI = Obj.computeConditionalMultiVariateMutualInfo(agent1, agent2, np.expand_dims(target, axis = 0).T)
						multVarMI = Obj.computeMultiVariateMutualInfo(agent1, agent2)

						results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)] = {}
						results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)]['condMultVarMI'] = condMultVarMI
						results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)]['multVarMI'] = multVarMI
						results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)]['coinformation'] = condMultVarMI - multVarMI  #### a.k.a interaction information, net synergy, and integration					

					# print(results)
					# print(Obj.resultFolder + Obj.dataFolders[simulationSettings[settingIndex]][Obj.dataFolders[simulationSettings[settingIndex]].find('_si/') + 4 : \
					# 	len(Obj.dataFolders[simulationSettings[settingIndex]])] + '/', seed)
					# sys.exit()
					Obj.saveResults(Obj.resultFolder + Obj.dataFolders[simulationSettings[settingIndex]][Obj.dataFolders[simulationSettings[settingIndex]].find('_si/') + 4 : \
						len(Obj.dataFolders[simulationSettings[settingIndex]])] + '/', seed, results)

		simResultsDirs = list(os.listdir(Obj.resultFolder))				
		if '.DS_Store' in simResultsDirs:
			simResultsDirs.remove('.DS_Store')
		
		for resultDirIndex in range(len(simResultsDirs)):
			# print(simResultsDirs[resultDirIndex], '  ', Obj.resultFolder + simResultsDirs[resultDirIndex])
			if whichSetting == 2:
				if resultDirIndex == 0:
					individual = Obj.prepareDataForAnalysis(Obj.resultFolder + simResultsDirs[resultDirIndex])
				elif resultDirIndex == 1:
					group = Obj.prepareDataForAnalysis(Obj.resultFolder + simResultsDirs[resultDirIndex])
				elif resultDirIndex == 2:
					joint = Obj.prepareDataForAnalysis(Obj.resultFolder + simResultsDirs[resultDirIndex])
		# print(individual.shape)
		# print(group.shape)
		# print(joint.shape)

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

		Obj.performFriedman_n_PosthocWilcoxonTest(condMultVarMI, 'Multivariate Conditional Mutual Information', 'Z-Scored Multivariate Conditional Mutual Information' \
			if Obj.whichNormalization == 1 else '[0 ..1] Scaled Multivariate Conditional Mutual Information')
		Obj.performFriedman_n_PosthocWilcoxonTest(multVarMI, 'Multivariate Mutual Information', 'Z-Scored Multivariate Mutual Information' \
			if Obj.whichNormalization == 1 else '[0 ..1] Scaled Multivariate Mutual Information')
		Obj.performFriedman_n_PosthocWilcoxonTest(coinformation, 'Net-Synergy', 'Z-Scored Net-Synergy' \
			if Obj.whichNormalization == 1 else '[0 ..1] Scaled Multivariate Net-Synergy')

		Obj.shutdownJVM()				



		# for seed in Obj.acceptedSeeds:
		# 	dir = f'data/phil_trans_si/1d_2n_exc-0.1_zfill_rp-3_switch/seed_{str(seed).zfill(3)}'		
		# 	perf, sim_perfs, evo, sim, data_record_list, sim_idx = run_simulation_from_dir(dir = dir, generation = Obj.generation)
		# 	# print(sim_perfs)
		# 	# sys.exit()
		# 	print(f'======  @ seed_{str(seed).zfill(3)}')
		# 	# print('# of Simulations =  ', len(data_record_list))
		# 	results = {}		
		# 	simIndex = sim_perfs.index(max(sim_perfs))
		# 	print('Simulation # ', (simIndex + 1), '   ', 'sim' + str())				
		# 	if 'sim' + str(simIndex + 1) not in results:
		# 		results['sim' + str(simIndex + 1)] = {}
		# 	for trialIndex in range(len(data_record_list[simIndex]['agents_brain_output'])):
		# 		print('Trial # ', (trialIndex + 1))
		# 		agent1, agent2, target = Obj.returnAgentsTargetData(data_record_list[simIndex], Obj.includedNodes, trialIndex)			
		# 		# print(agent1.shape, '  ', agent2.shape, '  ', target.shape)
		# 		condMultVarMI = Obj.computeConditionalMultiVariateMutualInfo(agent1, agent2, np.expand_dims(target, axis = 0).T)
		# 		multVarMI = Obj.computeMultiVariateMutualInfo(agent1, agent2)

		# 		results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)] = {}
		# 		results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)]['condMultVarMI'] = condMultVarMI
		# 		results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)]['multVarMI'] = multVarMI
		# 		results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)]['coinformation'] = condMultVarMI - multVarMI  #### a.k.a interaction information, net synergy, and integration				

		# 	# for simIndex in range(len(data_record_list)):
		# 	# 	print('Simulation # ', (simIndex + 1), '   ', 'sim' + str())				
		# 	# 	if 'sim' + str(simIndex + 1) not in results:
		# 	# 		results['sim' + str(simIndex + 1)] = {}
		# 	# 	for trialIndex in range(len(data_record_list[simIndex]['agents_brain_output'])):
		# 	# 		print('Trial # ', (trialIndex + 1))
		# 	# 		agent1, agent2, target = Obj.returnAgentsTargetData(data_record_list[simIndex], Obj.includedNodes, trialIndex)			
		# 	# 		# print(agent1.shape, '  ', agent2.shape, '  ', target.shape)
		# 	# 		condMultVarMI = Obj.computeConditionalMultiVariateMutualInfo(agent1, agent2, np.expand_dims(target, axis = 0).T)
		# 	# 		multVarMI = Obj.computeMultiVariateMutualInfo(agent1, agent2)
		# 	# 		# print(condMultVarMI, '  ', multVarMI)

		# 	# 		results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)] = {}
		# 	# 		results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)]['condMultVarMI'] = condMultVarMI
		# 	# 		results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)]['multVarMI'] = multVarMI
		# 	# 		results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)]['coinformation'] = condMultVarMI - multVarMI  #### a.k.a interaction information, net synergy, and integration

		# 	Obj.saveResults(resultFolder, f'seed_{str(seed).zfill(3)}', results)

		


		# Obj.checkDataNormality(condMultVarMI.flatten().tolist(), 'Multivariate Conditional Mutual Information')
		# Obj.checkDataNormality(multVarMI.flatten().tolist(), 'Multivariate Mutual Information')
		# Obj.checkDataNormality(coinformation.flatten().tolist(), 'Net-Synergy')	
		
	except Exception as e:
		print('@ synrgy.py main() :  ', e)
		sys.exit()