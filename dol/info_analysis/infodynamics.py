import os
import jpype as jp

def start_JVM():
    jarLocation = "infodynamics.jar"

    if (not(os.path.isfile(jarLocation))):
        exit("infodynamics.jar not found (expected at " + os.path.abspath(jarLocation) + ") - are you running from demos/python?")			
    jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation, convertStrings = False)   # convertStrings = False to silence the Warning while starting JVM 						

def shutdown_JVM():
    jp.shutdownJVM()

start_JVM()

# init JP
jp_kraskov_pkg = jp.JPackage("infodynamics.measures.continuous.kraskov")

# MI
multivar_mi_calc = jp_kraskov_pkg.MutualInfoCalculatorMultiVariateKraskov1()            
multivar_mi_calc.setProperty("NOISE_LEVEL_TO_ADD", "0") # no noise for reproducibility

# Cond. MI
multivar_cond_mi_calc = jp_kraskov_pkg.ConditionalMutualInfoCalculatorMultiVariateKraskov1()
multivar_cond_mi_calc.setProperty("NOISE_LEVEL_TO_ADD", "0") # no noise for reproducibility

def compute_mi(sub_matrix_a, sub_matrix_b):
    """Matrices should have the same num of rows but they may differ in num of columns

    Args:
        matrix_a (ndarray): [description]
        matrix_b (ndarray): [description]
    """        

    multivar_mi_calc.initialise(sub_matrix_a.shape[1], sub_matrix_b.shape[1])		

    multivar_mi_calc.setObservations(
        jp.JArray(jp.JDouble, 2)(sub_matrix_a), 
        jp.JArray(jp.JDouble, 2)(sub_matrix_b)
    )
    multivar_mi = multivar_mi_calc.computeAverageLocalOfObservations()

    return multivar_mi

def compute_cond_mi(sub_matrix_a, sub_matrix_b, conditioning_matrix):
    """Matrices should have the same num of rows but they may differ in num of columns

    Args:
        matrix_a (ndarray): [description]
        matrix_b (ndarray): [description]
    """        

    multivar_cond_mi_calc.initialise(sub_matrix_a.shape[1], sub_matrix_b.shape[1], 1)		

    multivar_cond_mi_calc.setObservations(
        jp.JArray(jp.JDouble, 2)(sub_matrix_a), 
        jp.JArray(jp.JDouble, 2)(sub_matrix_b),
        jp.JArray(jp.JDouble, 2)(conditioning_matrix)
    )
    multivar_cond_mi = multivar_cond_mi_calc.computeAverageLocalOfObservations()

    return multivar_cond_mi