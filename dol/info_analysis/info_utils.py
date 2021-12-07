import numpy as np

def interpretObservedEffectSize(effectSize, whichOne):
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

def showDescriptiveStatistics(data, whichOne):
    print('M-' + whichOne, ' = ', np.mean(data), ' SD-' + whichOne, ' = ', np.std(data), '  Mdn-' + whichOne, ' = ', np.median(data), \
        '  CI_95%-' + whichOne + ' = ', [np.percentile(data, 2.5), np.percentile(data, 97.5)])
