#! /usr/bin/env python 3


import sys
import os
import numpy as np
import pickle

import jpype as jp

class infoAnalysis:
	def __init__(self):
		try:
			self.initiJVM()

			self.generation = 5000
			self.acceptedSeeds = [2, 3, 6, 10, 11, 12, 13, 14, 15, 16, 18, 19]  ## This is the list of valid seeds

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
				os.mkdir(addr)

			if not os.path.exists(addr + seedName):
				os.mkdir(addr + seedName)				

			with open(addr + seedName + '/resultsDic.pickle', 'wb') as handle:
				pickle.dump(resultsDic, handle, protocol = pickle.HIGHEST_PROTOCOL)
				handle.close()

		except Exception as e:
			print('@ saveResults() :  ', e)
			sys.exit()

	def showResults(self, addr):
		try:
			for seed in self.acceptedSeeds:
				with open(addr + f'seed_{str(seed).zfill(3)}' + '/resultsDic.pickle', 'rb') as handle:
					data = pickle.load(handle)						
					handle.close()	
					print('====   ', f'seed_{str(seed).zfill(3)}')
					print(data)						
		except Exception as e:
			print('@ showResults() :  ', e)
			sys.exit()

