import os
import jpype as jp

def startJVM():
    jarLocation = "infodynamics.jar"

    if (not(os.path.isfile(jarLocation))):
        exit("infodynamics.jar not found (expected at " + os.path.abspath(jarLocation) + ") - are you running from demos/python?")			
    jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation, convertStrings = False)   # convertStrings = False to silence the Warning while starting JVM 						

def shutdownJVM():
    jp.shutdownJVM()
