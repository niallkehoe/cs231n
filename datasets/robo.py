from roboflow import Roboflow

rf = Roboflow(api_key="cUuR6Zpj8mZ1ZRFErnYo")

# 1. ComputerMonitor: https://universe.roboflow.com/n-j7ohx/computer-monitor-0cbhd/dataset/1
# project = rf.workspace("n-j7ohx").project("computer-monitor-0cbhd")
# version = project.version(1)
# dataset = version.download("yolov11")
                

# 2. MonitorsTvsPcMonitorsEtc: https://universe.roboflow.com/energy-chaser/monitors-tvs-pc-monitors-etc
# project = rf.workspace("energy-chaser").project("monitors-tvs-pc-monitors-etc")
# version = project.version(2)
# dataset = version.download("yolov11")
                
# 3. OfficeMonitor: https://universe.roboflow.com/4-52p2c/office-monitor-r7oge
# project = rf.workspace("4-52p2c").project("office-monitor-r7oge")
# version = project.version(1)
# dataset = version.download("yolov11")
                
# --- Segmentation datasets ---

# 4. screen-1: https://universe.roboflow.com/pavement-wwadi/screen-7i6h8
# project = rf.workspace("n-j7ohx").project("screen-7i6h8-9nqjc")
# version = project.version(1)
# dataset = version.download("yolov11")
                
# 5. screens-segmentation: https://universe.roboflow.com/myworkspace-mvnb3/screens-segmentation
project = rf.workspace("myworkspace-mvnb3").project("screens-segmentation")
version = project.version(5)
dataset = version.download("yolov11")
