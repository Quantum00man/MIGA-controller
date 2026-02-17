import ctypes
import os
from ctypes import *

try:
    vkdaq_home = os.getenv('VKDAQ_HOME',"/opt/vkdaq/lib")
    if not vkdaq_home:
        raise OSError("Environment variable VKDAQ_HOME is not set")
    dll_path = os.path.abspath(os.path.join(vkdaq_home,  'libvkdaq.so'))
    vkdaq = ctypes.CDLL(dll_path)
except OSError as e:
    print(f"Failed to load DLL: {e}")


#Example of using char tasks[] data type   Satrt
    # buffer_size = 100
    # buffer = create_string_buffer(buffer_size)
    # result = VkDaqGetTasks(buffer, buffer_size)
    # readtask = buffer.value
    # print(readtask.decode('utf-8'))
#Example of using char tasks[] data type   End



# int32_t VkDaqGetTasks(char tasks[], int32_t size);
VkDaqGetTasks=vkdaq.VkDaqGetTasks
VkDaqGetTasks.argtypes=[POINTER(c_char),c_int]
VkDaqGetTasks.restypes =c_int

# int32_t VkDaqCreateTask(const char* task);
VkDaqCreateTask=vkdaq.VkDaqCreateTask
VkDaqCreateTask.argtypes=[POINTER(c_char)]
VkDaqCreateTask.restypes =c_int

# int32_t VkDaqClearTask(const char* task);
VkDaqClearTask=vkdaq.VkDaqClearTask
VkDaqClearTask.argtypes=[POINTER(c_char)]
VkDaqClearTask.restypes =c_int

# int32_t VkDaqStartTask(const char* task);
VkDaqStartTask=vkdaq.VkDaqStartTask
VkDaqStartTask.argtypes=[POINTER(c_char)]
VkDaqStartTask.restypes =c_int

# int32_t VkDaqStopTask(const char* task);
VkDaqStopTask=vkdaq.VkDaqStopTask
VkDaqStopTask.argtypes=[POINTER(c_char)]
VkDaqStopTask.restypes =c_int


# int32_t VkDaqGetTaskAttribute(const char* task, const char* attrName, char attrValue[], int32_t size);
VkDaqGetTaskAttribute=vkdaq.VkDaqGetTaskAttribute
VkDaqGetTaskAttribute.argtypes=[POINTER(c_char),POINTER(c_char),POINTER(c_char),c_int]
VkDaqGetTaskAttribute.restypes =c_int

# int32_t VkDaqSetTaskAttribute(const char* task, const char* attrName, const char* attrValue);
VkDaqSetTaskAttribute=vkdaq.VkDaqSetTaskAttribute
VkDaqSetTaskAttribute.argtypes=[POINTER(c_char),POINTER(c_char),POINTER(c_char)]
VkDaqSetTaskAttribute.restypes =c_int


# int32_t VkDaqGetTaskData(const char* task, double dat[], int32_t sampsPerChan, int32_t fillmode, double timeout);
VkDaqGetTaskData=vkdaq.VkDaqGetTaskData
VkDaqGetTaskData.argtypes=[POINTER(c_char),POINTER(c_double),c_int,c_int,c_double]
VkDaqGetTaskData.restypes =c_int

# int32_t VkDaqSetTaskData(const char* task, const double dat[], int32_t sampsPerChan, int32_t fillmode, int32_t autoStart, double timeout);
VkDaqSetTaskData=vkdaq.VkDaqSetTaskData
VkDaqSetTaskData.argtypes=[POINTER(c_char),POINTER(c_double),c_int,c_int,c_int,c_double]
VkDaqSetTaskData.restypes =c_int


# int32_t VkDaqAddDevice(const char* dev);
VkDaqAddDevice=vkdaq.VkDaqAddDevice
VkDaqAddDevice.argtypes=[POINTER(c_char)]
VkDaqAddDevice.restypes =c_int

# int32_t VkDaqGetDevices(char devsAddress[], char devsName[], int32_t size);
VkDaqGetDevices=vkdaq.VkDaqGetDevices
VkDaqGetDevices.argtypes=[POINTER(c_char),POINTER(c_char),c_int]
VkDaqGetDevices.restypes =c_int

# int32_t VkDaqGetChannels(const char* dev, char channels[], int32_t size);
VkDaqGetChannels=vkdaq.VkDaqGetChannels
VkDaqGetChannels.argtypes=[POINTER(c_char),POINTER(c_char),c_int]
VkDaqGetChannels.restypes =c_int

# int32_t VkDaqGetDeviceAttribute(const char* dev, const char* attrName, char attrValue[], int32_t size);
VkDaqGetDeviceAttribute=vkdaq.VkDaqGetDeviceAttribute
VkDaqGetDeviceAttribute.argtypes=[POINTER(c_char),POINTER(c_char),POINTER(c_char),c_int]
VkDaqGetDeviceAttribute.restypes =c_int

# int32_t VkDaqSetDeviceAttribute(const char* dev, const char* attrName, const char* attrValue);
VkDaqSetDeviceAttribute=vkdaq.VkDaqSetDeviceAttribute
VkDaqSetDeviceAttribute.argtypes=[POINTER(c_char),POINTER(c_char),POINTER(c_char)]
VkDaqSetDeviceAttribute.restypes =c_int

# int32_t VkDaqGetChannelAttribute(const char* chan, const char* attrName, char attrValue[], int32_t size);
VkDaqGetChannelAttribute=vkdaq.VkDaqGetChannelAttribute
VkDaqGetChannelAttribute.argtypes=[POINTER(c_char),POINTER(c_char),POINTER(c_char),c_int]
VkDaqGetChannelAttribute.restypes =c_int

# int32_t VkDaqSetChannelAttribute(const char* chan, const char* attrName, const char* attrValue);
VkDaqSetChannelAttribute=vkdaq.VkDaqSetChannelAttribute
VkDaqSetChannelAttribute.argtypes=[POINTER(c_char),POINTER(c_char),POINTER(c_char)]
VkDaqSetChannelAttribute.restypes =c_int


# int32_t VkDaqCreateAIVoltageChan(const char* task, const char* chans, const char* reserved1, int32_t terminalConfig, double minVal, double maxVal, int32_t units, const char* reserved2);
VkDaqCreateAIVoltageChan=vkdaq.VkDaqCreateAIVoltageChan
VkDaqCreateAIVoltageChan.argtypes=[POINTER(c_char),POINTER(c_char),POINTER(c_char),c_int,c_double,c_double,c_int,POINTER(c_char)]
VkDaqCreateAIVoltageChan.restypes =c_int

# int32_t VkDaqCreateAICurrentChan(const char* task, const char* chans, const char* reserved1, int32_t terminalConfig, double minVal, double maxVal, int32_t units, const char* reserved2);
VkDaqCreateAICurrentChan=vkdaq.VkDaqCreateAICurrentChan
VkDaqCreateAICurrentChan.argtypes=[POINTER(c_char),POINTER(c_char),POINTER(c_char),c_int,c_double,c_double,c_int,POINTER(c_char)]
VkDaqCreateAICurrentChan.restypes =c_int

# int32_t VkDaqCreateAIAccelChan(const char* task, const char* chans, const char* reserved1, int32_t terminalConfig, double minVal, double maxVal, int32_t units, double sensitivity, int32_t signalType, const char* reserved2);
VkDaqCreateAIAccelChan=vkdaq.VkDaqCreateAIAccelChan
VkDaqCreateAIAccelChan.argtypes=[POINTER(c_char),POINTER(c_char),POINTER(c_char),c_int,c_double,c_double,c_int,c_double,c_int,POINTER(c_char)]
VkDaqCreateAIAccelChan.restypes =c_int

# int32_t VkDaqCreateDIChan(const char* task, const char* chans, const char* reserved1, uint32_t mask);
VkDaqCreateDIChan=vkdaq.VkDaqCreateDIChan
VkDaqCreateDIChan.argtypes=[POINTER(c_char),POINTER(c_char),POINTER(c_char),c_int]
VkDaqCreateDIChan.restypes =c_int

# int32_t VkDaqCreateMIChan(const char* task, const char* chans, const char* reserved1, const char* source, uint32_t maxVal);
VkDaqCreateMIChan=vkdaq.VkDaqCreateMIChan
VkDaqCreateMIChan.argtypes=[POINTER(c_char),POINTER(c_char),POINTER(c_char),POINTER(c_char),c_int]
VkDaqCreateMIChan.restypes =c_int


# 	LIBVKDAQ_API int32_t VkDaqCreateAOVoltageChan(const char* task, const char* chans, const char* reserved1, double minVal, double maxVal, int32_t units, const char* reserved2);
VkDaqCreateAOVoltageChan=vkdaq.VkDaqCreateAOVoltageChan
VkDaqCreateAOVoltageChan.argtypes=[POINTER(c_char),POINTER(c_char),POINTER(c_char),c_double,c_double,c_int,POINTER(c_char)]
VkDaqCreateAOVoltageChan.restypes =c_int


# int32_t VkDaqCfgSampClkTiming(const char* task, int32_t source, double samplingFrequency, int32_t reserved, int32_t sampleMode, int32_t sampsPerChanToAcquire);
VkDaqCfgSampClkTiming=vkdaq.VkDaqCfgSampClkTiming
VkDaqCfgSampClkTiming.argtypes=[POINTER(c_char),c_int,c_double,c_int,c_int,c_int]
VkDaqCfgSampClkTiming.restypes =c_int


# int32_t VkDaqCfgDigEdgeRefTrig(const char* task, const char* triggerSource, int32_t triggerEdge, uint32_t reserved);
VkDaqCfgDigEdgeRefTrig=vkdaq.VkDaqCfgDigEdgeRefTrig
VkDaqCfgDigEdgeRefTrig.argtypes=[POINTER(c_char),POINTER(c_char),c_int,c_int]
VkDaqCfgDigEdgeRefTrig.restypes =c_int


# int32_t VkDaqCfgAnlgEdgeRefTrig(const char* task, const char* triggerSource, int32_t triggerEdge, double triggerLevel, uint32_t reserved);
VkDaqCfgAnlgEdgeRefTrig=vkdaq.VkDaqCfgAnlgEdgeRefTrig
VkDaqCfgAnlgEdgeRefTrig.argtypes=[POINTER(c_char),POINTER(c_char),c_int,c_double,c_int]
VkDaqCfgAnlgEdgeRefTrig.restypes =c_int

# int32_t VkDaqAssistantDisplay(const char* tray, const char* toast);
VkDaqAssistantDisplay=vkdaq.VkDaqAssistantDisplay
VkDaqAssistantDisplay.argtypes=[POINTER(c_char),POINTER(c_char)]
VkDaqAssistantDisplay.restypes =c_int







VkDaqGetLastErrorInfo=vkdaq.VkDaqGetLastErrorInfo
VkDaqGetLastErrorInfo=c_char

VkDaq_Val_SingleEnded = 0
VkDaq_Val_Differential = 1

#Values for Vkdaq_AI_Voltage_Units
VkDaq_Val_Volts = 0
VkDaq_Val_Millivolt = 1
VkDaq_Val_CustomUnit = 2

#Values for Vkdaq_AI_Current_Units
VkDaq_Val_Amperes = 0
VkDaq_Val_Milliampere = 1
# VkDaq_Val_CustomUnit = 2

#Values for Vkdaq_AI_Accel_Charge_Sensitivity_Units
VkDaq_Val_AccelUnit_g = 0
VkDaq_Val_MetersPerSecondSquared = 1
VkDaq_Val_InchesPerSecondSquared = 2
VkDaq_Val_FromCustomScale = 3

DAQmx_Val_SignalType_IEPE = 0
DAQmx_Val_SignalType_PE = 1

VkDaq_Val_Falling = 0
VkDaq_Val_Rising = 1
VkDaq_Val_ClkSrc_OnBoardClk = 0
VkDaq_Val_ClkSrc_SyncPluse = 1
VkDaq_Val_ClkSrc_MasterSlaves =2

VkDaq_Val_ContSamps = 0
VkDaq_Val_FiniteSamps = 1

VkDaq_Val_GroupByScanNumber = 0
VkDaq_Val_GroupByChannel = 1


