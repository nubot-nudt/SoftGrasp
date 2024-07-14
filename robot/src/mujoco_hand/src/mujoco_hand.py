#!/home/nubot-11/Data/anaconda3/envs/DAPG-1/bin/python
import mujoco_py
from scipy.spatial.transform import Rotation as R
import time
# import rospy
# from std_msgs.msg import Int32MultiArray
# from tf2_msgs.msg import TFMessage
# from geometry_msgs.msg import Point
# from geometry_msgs.msg import Pose
import tf
import pickle
import copy
import numpy as np
# from ur_msgs.msg import URSub



mj_path = '/path/to/mujoco200'
model_path = '/home/nubot-11/Data/code/hand/two_hand/src/mujoco_hand/src/hand_relocate_1.xml'

# rospy.init_node('mujoco_simulator')

Arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
speed_arr=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
vive_pos_bias = 0
base_matrix = np.identity(4)
vive_matrix = np.eye(4,4)
vive_base_matrix = np.identity(4)
vive_base_matrix[:,3] = [-0.16175, 0.0, -0.138, 1]

robot_position = []
robot_height = []
robot_pitch = []
fin_angle_real = np.empty(shape=(0,4))
in_angle_expect = np.empty(shape=(0,4))
tcp_pose = [0.00,-0.3,0.135,0,0,0]



# 加载模型和仿真器
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)

# 创建查看器对象并连接到仿真器
viewer = mujoco_py.MjViewer(sim)

sim.forward()

target_obj_sid = sim.model.site_name2id("target")
S_grasp_sid = sim.model.site_name2id('S_grasp')
obj_bid = sim.model.body_name2id('Object')
# 设置初始速度
for joint_name in sim.model.joint_names:
    joint_id = sim.model.joint_name2id(joint_name)
    sim.data.qvel[joint_id] = 0.0

# def InfoCallback(data):
#     i = 0
#     for i in range(min(len(data.data), len(Arr))):
#         Arr[i] = (data.data[i])*0.001
#         # Arr[i]=Arr[i]*0.01
#         # print("inspire_speed_cmd: %.3f,%.3f,%.3f,%.3f,%.3f,%.3f" %(Arr[6],Arr[7],speed_arr[8],Arr[9],Arr[10],Arr[11]))
#     file_path = "inspire_speed_cmd.txt"  # 定义文件路径

#     # with open(file_path, "a") as file:
#     #     file.write("inspire_speed_cmd: %.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n" % (Arr[6], Arr[7], Arr[8], Arr[9], Arr[10], Arr[11]))

   
# def URControllerCallback(data):
#     global tcp_pose
#     tcp_pose = data.Pose_signals
    

    
demos = pickle.load(open('hand-v1_demos.pickle', 'rb'))

for path in demos:
        
        print("path_type:",type(path))
        init_state_dict=(path['init_state_dict']) 
        qp = init_state_dict['qpos']
        qv = init_state_dict['qvel']
        # print("qp.shape:", qp.shape)
        # print("qv.shape:", qv.shape)
        # self.sim.data.qpos = qp
        obj_pos = init_state_dict['obj_pos']
        target_pos = init_state_dict['target_pos']
        print("obj_pos:", obj_pos)
        # obj_pos[1] = obj_pos[1]+0.1
        print("obj_pos:", obj_pos)

        # sim.set_state(qp)
        sim.model.body_pos[obj_bid] = obj_pos
        sim.model.site_pos[target_obj_sid] = target_pos
  
        actions = path['actions']

        for t in range(actions.shape[0]):
            # print("actions:", actions[t])
            
            sim.data.qpos[sim.model.joint_name2id("FFJ2")] = actions[t][6]
            sim.data.qpos[sim.model.joint_name2id("MFJ2")] = actions[t][8]
            sim.data.qpos[sim.model.joint_name2id("RFJ2")] = actions[t][10]
            sim.data.qpos[sim.model.joint_name2id("LFJ2")] = actions[t][12]
            sim.data.qpos[sim.model.joint_name2id("THJ3")] = actions[t][14]
            sim.data.qpos[sim.model.joint_name2id("THJ2")] = actions[t][16]

            arm_link_id = sim.model.body_name2id('arm_Link')

            sim.model.body_pos[arm_link_id][:] = actions[t][0:3]
            # # print('target_tcp_pose',tcp_pose)

            # # target_tcp_pose[3:6] 包含轴角表示法中的(rx, ry, rz)
            rot_hand = R.from_rotvec( actions[t][3:6], degrees=False).as_matrix()
            rot_hand = R.from_matrix(rot_hand)
            quat_hand = R.as_quat(rot_hand)
            
            sim.model.body_quat[arm_link_id] = [quat_hand[3],quat_hand[0],quat_hand[1],quat_hand[2]]
            # sim.data.qpos[sim.model.joint_name2id("ARRx")] = euler_tcp_pose[3]
            # sim.data.qpos[sim.model.joint_name2id("ARRy")] = euler_tcp_pose[4]
            # sim.data.qpos[sim.model.joint_name2id("ARRz")] = euler_tcp_pose[5]

            time.sleep(0.02) 

            sim.step()

            # 更新查看器并渲染仿真界面
            viewer.render()
    
            


# rospy.Subscriber('dexmo_hand_joint', Int32MultiArray, InfoCallback)
# rospy.Subscriber('/URSub', URSub, URControllerCallback)

# for i in range(100000):
        
#         # print("inspire_cmd: %.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n" % (Arr[0], Arr[1], Arr[2], Arr[3], Arr[4], Arr[5]))
        
#         sim.data.qpos[sim.model.joint_name2id("FFJ2")] = (1-Arr[3])
#         sim.data.qpos[sim.model.joint_name2id("MFJ2")] = (1-Arr[2])
#         sim.data.qpos[sim.model.joint_name2id("RFJ2")] = (1-Arr[1])
#         sim.data.qpos[sim.model.joint_name2id("LFJ2")] = (1-Arr[0])
#         sim.data.qpos[sim.model.joint_name2id("THJ3")] = (1-Arr[4])
#         sim.data.qpos[sim.model.joint_name2id("THJ2")] = (1-Arr[5])

#         # sim.data.qpos[sim.model.joint_name2id("ARTx")] = tcp_pose[0]
#         # sim.data.qpos[sim.model.joint_name2id("ARTy")] = tcp_pose[1]
#         # sim.data.qpos[sim.model.joint_name2id("ARTz")] = tcp_pose[2]
        
#         arm_link_id = sim.model.body_name2id('arm_Link')

#         sim.model.body_pos[arm_link_id][:] = tcp_pose[0:3]
#         # # print('target_tcp_pose',tcp_pose)

#         # # target_tcp_pose[3:6] 包含轴角表示法中的(rx, ry, rz)
        
       
#         print('before_tcp_pose:',tcp_pose)
#         rotation = R.from_rotvec(tcp_pose[3:6])
#         euler_tcp_pose = list(tcp_pose)  # 将元组转换为列表
#         euler_tcp_pose[3:6] = rotation.as_euler('zyx', degrees=False)
#         print('after_tcp_pose:',euler_tcp_pose)
#         # tcp_pose = tuple(tcp_pose)  # 将修改后的列表转换回元组
#         rot_vec = np.array(tcp_pose[3:6])
#         quat = R.from_rotvec(rot_vec).as_quat()
#         print(quat)

#         sim.model.body_quat[arm_link_id] = [quat[3],quat[0],quat[1],quat[2]]
#         # sim.data.qpos[sim.model.joint_name2id("ARRx")] = euler_tcp_pose[3]
#         # sim.data.qpos[sim.model.joint_name2id("ARRy")] = euler_tcp_pose[4]
#         # sim.data.qpos[sim.model.joint_name2id("ARRz")] = euler_tcp_pose[5]

#         time.sleep(0.02) 

#         sim.step()

#         # 更新查看器并渲染仿真界面
#         viewer.render()
        
  
# rospy.spin()