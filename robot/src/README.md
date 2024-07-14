# ROS node to connect Inspire & DEXMO hand  

## 一、Complie

```
mkdir inspire_dexmo_hand  
cd inspire_dexmo_hand  
mkdir src  
cd src 
git clone git@gitee.com:skywalker1941/inspire_dexmo_hand.git
cd ..  
catkin_make  
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
sourece devel/setup.bash
```  
---  

## 二、Run 

```
cd Data/code/hand/two_hand
```

## 主手程序：dexmo_hand  
### 1. run dexmo_server

>cd src/dexmo_server

>sudo ./dexmo_server_1.1.6

### 2. run dexmo ros node

>source devel/setup.bash

>roslaunch dexmo_hand dexmo_hand.launch

## 主手仿真：mujoco_hand 

```
source devel/setup.bash
cd src/mujoco_hand/src

chmod +x mujoco_hand.py
roslaunch mujoco_hand mujoco_hand.launch
```

## 从手程序：inspire_hand  
### run inspire ros node
```
source devel/setup.bash
roslaunch inspire_hand inspire_hand_control.launch
```
```
roslaunch dexmo_hand dexmo_hand_ubuntu.launch
roslaunch inspire_hand hand_control.launch test_flag:=1
roslaunch inspire_hand inspire_hand_control_ubuntu.launch
```
## viev程序: vive_ur3
```
conda activate dexmv-1

source devel/setup.bash
roslaunch vive_ur3 vive_ur3.launch
```
或  
`roslaunch vive_ur3 vive_ur3_1.launch`  
在`/config/vive_ur3.yaml` 中修改参数  
记得修改  
`vive_ur3_node.py` 第一行环境路径以及文件执行权限
## ur3程序：ur3_control
```
urcontrol
ursim

source devel/setup.bash
rosrun ur_control ur_control.py
```


## UR虚拟
该文件夹为ursim（作用是代替实物ur机械臂的urcap，验证算法），下载链接为https://www.universal-robots.com/download/software-cb-series/simulator-linux/offline-simulator-cb3-linux-ursim-3158/


解压之后cd ursim-3.15.8.106339，修改该文件line70，将 libcurl3 改为 libcurl4:i386，注释掉line86~97


### ursim仿真软件安装：

```
cd ursim-3.15.8.106339

. /install.sh

可能会报错，查一下缺的包，常见的问题是java版本不匹配
https://github.com/cf-dtx/CoppeliaURSim

Yet, and for the URSim to launch you will need to shift to Java 8. First install the version 8:

If you have never installed java in your machine, when you write theese commands, you should get the following output:

$ sudo apt update
$ sudo apt install openjdk-8-jdk openjdk-8-jre
$ java -version
openjdk version "1.8.0_252"
OpenJDK Runtime Environment (build 1.8.0_252-8u252-b09-1ubuntu1-b09)
OpenJDK 64-Bit Server VM (build 25.252-b09, mixed mode)

choose right java version:
sudo update-alternatives --config java
2
(java-8-openjdk-amd64)
```

### ursim仿真软件启动：

```
cd ursim-3.15.8.106339
sudo ./starturcontrol.sh     #该命令每次电脑重启时运行一次就可以了
./start-ursim.sh UR3   #启动ur3的ursim
```

in ~/.bashrc add like:

alias ursim="/home/nubot-11/Data/5_ursim_CB3/ursim-3.15.8.106339/start-ursim.sh"
alias urcontrol="cd /home/nubot-11/Data/5_ursim_CB3/ursim-3.15.8.106339/; sudo ./starturcontrol.sh"

source ~/.bashrc
then you can just input ursim or urcontrol in terminal to run.


随后可以通过ur_socket对该机械臂进行控制，具体代码见ursim_example.py

HOST = "127.0.0.1"

PORT = 30003 

而在与实物进行连接时需要注意将HOST修改为实际配置的结果

## 手眼标定程序
### realsense相机启动测试(测试后关闭)
realsense相机驱动：├── reslsense-d435i

realsense相机需要安装在3.0 USB口
```
roslaunch realsense2_camera demo_pointcloud.launch 
roslaunch realsense2_camera rs_camera.launch
realsense-viewer
```
### 手眼标定程序（眼在手上）
通过ursim或者ur实物的socket方式获取末端坐标（法兰中心/tcp，要确定一下），发布话题"/arm_pose"
```
source devel/setup.bash
roslaunch handeye-calib strat_moduanpose.launch
```

启动realsense（rs_camera.launch）,和aruco_ros单标定板的标定程序，发布话题"/aruco_single/pose"
```
cd Data/reslsense-d435i/
source devel/setup.bash
<!-- roslaunch realsense2_camera rs_camera.launch -->
roslaunch realsense2_camera demo_pointcloud.launch

source devel/setup.bash
roslaunch handeye-calib aruco_start_realsense_sdk_target.launch
roslaunch handeye-calib aruco_start_realsense_sdk_object.launch
roslaunch handeye-calib double.launch
```
接收上述两个话题的消息，使用OpenCV的4种标定方法进行数据记录和计算，得到eye_on_hand标定结果（相机和手之间的坐标变换关系），并通过标定板相较于基坐标系之间的数据均值进行结果对比
```
conda activate DAPG-1
roslaunch handeye-calib online_hand_on_eye_calib.launch
```
如眼在手上可选择end_link->marker某一算法输出结果为最终结果

眼在手外的代码启动流程：
```
roslaunch handeye-calib online_hand_to_eye_calib.launch
```

### 手眼标定测试程序（眼在手上）
输入：

1.aruco码在相机坐标系下的坐标 camera_frame->aruco_marker_frame

2.手眼标定结果base_link->camera_frame

输出：

1.aruco码在机械臂基坐标下的位置,base_link->aruco_maker_frame

配置launch文件`src/handeye-calib/launch/test/test_hand_on_eye_calib.launch`

```
    <arg   name="base_link"   default="/base_link" />
    <arg   name="end_link"   default="/link7_name" />

    <arg   name="base_link2camera_link" default="{'t':[0,0,0],'r':[0,0,0,0]}" />
    <arg   name="camera_link"   default="/camera_link" />
    <arg   name="marker_link"   default="/aruco_marker_frame"/>
```
- base_link， 机械臂基座tf名称
- end_link，机械臂末端坐标名称
- base_link2camera_link，机械臂基座和相机之间的位姿关系，手眼标定结果给出，t代表平移单位m，r代表旋转，四元数形式,顺序为qx,qy,qz,qw
- camera_link，aruco中的camera的frame id配置名字
- marker_link，aruco中marker的frame_id

使用下面的指令运行

```
source devel/setup.bash
roslaunch handeye-calib test_hand_on_eye_marker_base.launch
roslaunch handeye-calib test_hand_to_eye_marker_base.launch
```
程序运行基输出实时的base_link和aruco_marker_frame之间的关系.

```
result:/base_link->/aruco_marker_frame, [0.0, 0.0, 2.0],[0.0, 0.0, 0.0, 1.0]
```
https://chev.me/arucogen/
## 录包程序：
```
rosbag record -a
rosbag play 2020-04-11-09-11-04.bag
roslaunch exp_plot data_process_11.launch
roslaunch exp_plot data_bagplay.launch
roslaunch exp_plot data_replay.launch
```
## check：inspire_hand demo

```
source devel/setup.bash

roslaunch inspire_hand hand_control.launch test_flag:=1

source devel/setup.bash
         
rosservice call /inspire_hand/set_pos pos1 pos2 pos3 pos4 pos5 pos6 
设置六个驱动器位置------参数pos范围0-2000 

rosservice call /inspire_hand/set_angle angle1 angle2 angle3 angle4 angle5 angle6 
设置灵巧手角度------参数angle范围-1-1000

rosservice call /inspire_hand/set_force force1 force2 force3 force4 force5 force6 
设置力控阈值------参数force范围0-1000

rosservice call /inspire_hand/set_speed speed1 speed2 speed3 speed4 speed5 speed6 
设置速度------参数speed范围0-1000

rosservice call /inspire_hand/get_pos_act
读取驱动器实际的位置值

rosservice call /inspire_hand/get_angle_act
读取实际的角度值

rosservice call /inspire_hand/get_force_act
读取实际的受力

rosservice call /inspire_hand/get_pos_set
读取驱动器设置的位置值

rosservice call /inspire_hand/get_angle_set
读取设置的角度值

rosservice call /inspire_hand/get_force_set
读取设置的力控阈值
      
rosservice call /inspire_hand/get_error
读取故障信息

rosservice call /inspire_hand/get_status
读取状态信息

rosservice call /inspire_hand/get_temp
读取温度信息

rosservice call /inspire_hand/get_current
读取电流

rosservice call /inspire_hand/set_clear_error
清除错误
     
rosservice call /inspire_hand/set_default_speed speed1 speed2 speed3 speed4 speed5 speed6
设置上电速度------参数speedk范围0-1000

rosservice call /inspire_hand/set_default_force force1 force2 force3 force4 force5 force6
设置上电力控阈值------参数forcek范围0-1000

rosservice call /inspire_hand/set_save_flash
保存参数到FLASH

rosservice call /inspire_hand/set_force_clb
校准力传感器

rosservice call /inspire_hand set_run_action action
执行动作序列，参数action范围1～30

```

###
可以采用下面指令实时监控灵巧手的实际角度和力
需要重新打开一个新的终端:


```
  source devel/setup.bash
  rosrun inspire_hand handcontroltopicpublisher

  source devel/setup.bash
  rosrun inspire_hand handcontroltopicsubscriber
```