
from Phidget22.Devices.VoltageRatioInput import VoltageRatioInput
from Phidget22.VoltageRatioSensorType import VoltageRatioSensorType


class Slider(VoltageRatioInput):

    def __init__(self, port):
        VoltageRatioInput.__init__(self)   
        self.sensorValue = 0.     
        self.setIsHubPortDevice(True)
        self.setHubPort(port)
        self.setOnSensorChangeHandler(Slider.onSensorChange)
        self.openWaitForAttachment(5000)
        self.setSensorType(VoltageRatioSensorType.SENSOR_TYPE_1112)

    def onSensorChange(self, sensorValue, sensorUnit):        
        self.sensorValue = sensorValue        
