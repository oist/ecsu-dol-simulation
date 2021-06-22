#! /usr/bin/env python 3

import sys
import numpy as np
import os
from dol.run_from_dir import run_simulation_from_dir

from dol.synergyAnalysisClass import infoAnalysis


if __name__ == "__main__":
	try:
		resultFolder = './results_MultVarMI_CondMi_CoInfo/'
		
		Obj = infoAnalysis()
		
		for seed in Obj.acceptedSeeds:
			dir = f'data/phil_trans_si/1d_2n_exc-0.1_zfill_rp-3_switch/seed_{str(seed).zfill(3)}'		
			perf, sim_perfs, evo, sim, data_record_list, sim_idx = run_simulation_from_dir(dir = dir, generation = Obj.generation)
			print(f'======  @ seed_{str(seed).zfill(3)}')
			# print('# of Simulations =  ', len(data_record_list))
			results = {}		
			for simIndex in range(len(data_record_list)):
				print('Simulation # ', (simIndex + 1))
				if 'sim' + str(simIndex + 1) not in results:
					results['sim' + str(simIndex + 1)] = {}
				for trialIndex in range(len(data_record_list[simIndex]['agents_brain_output'])):
					print('Trial # ', (trialIndex + 1))
					agent1, agent2, target = Obj.returnAgentsTargetData(data_record_list[simIndex], ['agents_brain_input', 'agents_brain_state', 'agents_brain_output', 'target_position'], trialIndex)			
					# print(agent1.shape, '  ', agent2.shape, '  ', target.shape)
					condMultVarMI = Obj.computeConditionalMultiVariateMutualInfo(agent1, agent2, np.expand_dims(target, axis = 0).T)
					multVarMI = Obj.computeMultiVariateMutualInfo(agent1, agent2)
					# print(condMultVarMI, '  ', multVarMI)

					results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)] = {}
					results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)]['condMultVarMI'] = condMultVarMI
					results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)]['multVarMI'] = multVarMI
					results['sim' + str(simIndex + 1)]['trial' + str(trialIndex + 1)]['coinformation'] = condMultVarMI - multVarMI  #### a.k.a interaction information, net synergy, and integration

			Obj.saveResults(resultFolder, f'seed_{str(seed).zfill(3)}', results)

		condMultVarMI, multVarMI, coinformation = Obj.prepareDataForAnalysis(resultFolder)
		condMultVarMI = Obj.normalizeData(condMultVarMI, Obj.whichNormalization) 
		multVarMI = Obj.normalizeData(multVarMI, Obj.whichNormalization)
		coinformation = Obj.normalizeData(coinformation, Obj.whichNormalization)

		# Obj.checkDataNormality(condMultVarMI.flatten().tolist(), 'Multivariate Conditional Mutual Information')
		# Obj.checkDataNormality(multVarMI.flatten().tolist(), 'Multivariate Mutual Information')
		# Obj.checkDataNormality(coinformation.flatten().tolist(), 'Net-Synergy')

		Obj.performFriedman_n_PosthocWilcoxonTest(condMultVarMI, 'Multivariate Conditional Mutual Information', 'Z-Scored Multivariate Conditional Mutual Information' \
			if Obj.whichNormalization == 1 else '[0 ..1] Scaled Multivariate Conditional Mutual Information')
		Obj.performFriedman_n_PosthocWilcoxonTest(multVarMI, 'Multivariate Mutual Information', 'Z-Scored Multivariate Mutual Information' \
			if Obj.whichNormalization == 1 else '[0 ..1] Scaled Multivariate Mutual Information')
		Obj.performFriedman_n_PosthocWilcoxonTest(coinformation, 'Net-Synergy', 'Z-Scored Net-Synergy' \
			if Obj.whichNormalization == 1 else '[0 ..1] Scaled Multivariate Net-Synergy')

		Obj.shutdownJVM()			
		
	except Exception as e:
		print('@ synrgy.py main() :  ', e)
		sys.exit()