import numpy as np
import time
import copy
import rospy
import sys, select
import math

from std_msgs.msg import Int32MultiArray
# from ur_msgs.msg import URPub_pose
# from ur_msgs.msg import URSub
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty
import IPython
e = IPython.embed

class ImageRecorder:
    def __init__(self, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist'] #['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
        if init_node:
            rospy.init_node('image_recorder', anonymous=True)
     
            image_pub = rospy.Subscriber('/camera/color/image_raw',Image,self.image_Callback)
    
        time.sleep(0.5)
    def image_Callback(self,data):
        self.cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.timestr = "%.6f" %  data.header.stamp.to_sec()
        # 显示图像
    def image_cb(self, cam_name, data):
        setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough'))
        setattr(self, f'{cam_name}_secs', data.header.stamp.secs)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nsecs)
        # cv2.imwrite('/home/tonyzhao/Desktop/sample.jpg', cv_image)
        if self.is_debug:
            getattr(self, f'{cam_name}_timestamps').append(data.header.stamp.secs + data.header.stamp.secs * 1e-9)

    def image_cb_cam_high(self, data):
        cam_name = 'cam_high'
        return self.image_cb(cam_name, data)

    def image_cb_cam_low(self, data):
        cam_name = 'cam_low'
        return self.image_cb(cam_name, data)

    def image_cb_cam_left_wrist(self, data):
        cam_name = 'cam_left_wrist'
        return self.image_cb(cam_name, data)

    def image_cb_cam_right_wrist(self, data):
        cam_name = 'cam_right_wrist'
        return self.image_cb(cam_name, data)

    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)
        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f'{cam_name}_timestamps'))
            print(f'{cam_name} {image_freq=:.2f}')
        print()

class Recorder:
    def __init__(self,init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from sensor_msgs.msg import JointState

        self.secs = None
        self.nsecs = None
        self.qpos = None
        self.effort = None
        ######   
        self.inspire_data = None
        self.inspire_joint = None
        self.inspire_force = None
        self.ur_pose = None
        self.palm_pos = None
   
        self.pos = None
        self.tcp_pose = None

        # pickle_data
        self.hand_joint = None
        self.hand_joint_6 = None
        self.hand_joint_12 = None
        self.object_joint = None
        self.qpos_6 = None
        self.qpos_12 = None
        self.hand_qpos_6 = None
        self.hand_qpos_12 = None
        self.obj_pos = None
        self.target_pos = None
        self.qvel = None
        # action_data 
        self.action_12 = None
        self.action_6 = None

        self.observation = None


        self.is_debug = is_debug
        self.settings = self.saveTerminalSettings()

        if init_node:
            rospy.init_node('recorder', anonymous=True)

        inspire_sub = rospy.Subscriber('inspire_angle_force', Int32MultiArray, self.inspire2dexmo_Callback)

        dexmo_sub = rospy.Subscriber('/dexmo_hand_joint', Int32MultiArray, self.dexmo_Callback)
        # ur_tcp_sub = rospy.Subscriber('/UR_pose_Pub',URPub_pose,self.ur_callback)

        # vive_tcp_sub = rospy.Subscriber('/URSub',URSub,self.vive_tcpCallback)

        object_marker_base_pub = rospy.Subscriber('/ur3/object_marker_pose',PoseStamped,self.object_marker_baseCallback)
        target_marker_base_pub = rospy.Subscriber('/ur3/target_marker_pose',PoseStamped,self.target_marker_baseCallback)
        data_sample_Timer = rospy.Timer(rospy.Duration(1.0 / 10.0), self.Timer_Callback) 
       

        try:
            while(1):
                key = self.getKey(self.settings)
                self.keyboard_callback(key)
                if key == '\x03':
                    break

        except Exception as e:
            print(e)
    
        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
            self.gripper_command_timestamps = deque(maxlen=50)
        time.sleep(0.1)

    def saveTerminalSettings(self):
        if sys.platform == 'win32':
            return None
        return termios.tcgetattr(sys.stdin)

    def getKey(self,settings):
        if sys.platform == 'win32':
            # getwch() returns a string on Windows
            key = msvcrt.getwch()
        else:
            tty.setraw(sys.stdin.fileno())
            # sys.stdin.read() returns a string on Linux
            key = sys.stdin.read(1)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def inspire2dexmo_Callback(self, inspire_msg):
        self.inspire_data = inspire_msg.data
        self.inspire_data = np.array(self.inspire_data)
        # print('self.inspire_data',self.inspire_data) 
        self.inspire_data = self.inspire_data /1000 
        self.inspire_joint = self.inspire_data[0:6] 

        # self.inspire_force = self.inspire_data[-6:]
        self.hand_joint_12 = []
        for val in self.inspire_joint:
            self.hand_joint_12.append(val)
            self.hand_joint_12.append(0)

    def dexmo_Callback(self,data):
        i = 0
        for i in range(min(len(data.data), len(self.Arr))):
            self.Arr[i] = (data.data[i])*0.001
            self.dexmo =self.Arr

    # def ur_callback(self,TCP_Position):
    #     ##  TCP_Position.TCP_Position 的单位为毫米
    #     self.ur_pose[0] = TCP_Position.TCP_Position[0]
    #     self.ur_pose[1] = TCP_Position.TCP_Position[1]
    #     self.ur_pose[2] = TCP_Position.TCP_Position[2]
    #     self.ur_pose[3] = TCP_Position.TCP_Position[3]
    #     self.ur_pose[4] = TCP_Position.TCP_Position[4]
    #     self.ur_pose[5] = TCP_Position.TCP_Position[5]
    #     self.palm_pos = copy.deepcopy(self.ur_pose[0:3])
    #     # rotation = R.from_rotvec(self.ur_pose[3:6])
    #     # self.ur_pose[3:6] = rotation.as_euler('zyx', degrees=False)
    #     # print('self.ur_pose_as_euler',self.ur_pose)
    #     # print('self.palm_pos',self.palm_pos)
    #     self.palm_pos[0] = self.palm_pos[0]+0.1
    #     self.hand_joint_6 =self.inspire_data[0:6] 
    #     self.qpos_6 = np.concatenate([self.ur_pose,self.hand_joint_6, self.object_joint,self.pos,self.pos])
    #     self.hand_qpos_6 = self.qpos_6
    #     self.action_6 = np.concatenate([self.ur_pose,self.hand_joint_6, self.object_joint,self.pos])

    #     self.qpos_12 = np.concatenate([self.ur_pose,self.hand_joint_12, self.object_joint])
    #     # print('self.qpos_12',self.qpos_12.shape)
    #     # print('self.ur_pose',self.ur_pose)
    #     # print('self.hand_joint_12',self.hand_joint_12)
    #     # print('self.object_joint',self.object_joint)

    #     self.hand_qpos_12 = self.qpos_12
    #     # self.action_12 = np.concatenate([self.ur_pose,self.hand_joint_12],axis=None)
    #     # print('self.qpos',self.qpos)

    #     ### observation 
    #     # self.observation = np.concatenate([self.qpos_12[:-6],self.obj_pos,self.palm_pos,self.target_pos])
    #     # self.observation = np.concatenate([self.qpos_12[:-6],self.palm_pos-self.obj_pos,self.palm_pos-self.target_pos,self.obj_pos-self.target_pos])


    # def vive_tcpCallback(self,data):
    #     self.tcp_pose = copy.deepcopy(np.array(data.Pose_signals))

    def object_marker_baseCallback(self,data):

        self.object_marker_to_base_transform = {
            't': (data.pose.position.x, data.pose.position.y, data.pose.position.z),
            'r': (data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w)
        }
        
        self.obj_pos = (data.pose.position.x, data.pose.position.y, data.pose.position.z)
        print('self.obj_pos_1:',self.obj_pos)
        if self.obj_pos[0] == 0:
        # 保存当前的 obj_pos
            # self.obj_pos = self.obj_pos.copy()
            self.obj_pos = self.obj_pos_old
        # print('self.obj_pos:',self.obj_pos)
        else:
            self.obj_pos_old = self.obj_pos

    def target_marker_baseCallback(self,data2):
        self.target_marker_to_base_transform = {
            't': (data2.pose.position.x, data2.pose.position.y, data2.pose.position.z),
            'r': (data2.pose.orientation.x, data2.pose.orientation.y, data2.pose.orientation.z, data2.pose.orientation.w)
        }
        # print('target_marker_to_base_transform:',self.target_marker_to_base_transform)
        self.target_pos = (data2.pose.position.x , data2.pose.position.y, data2.pose.position.z)
        print('self.target_pos:',self.target_pos)

    def keyboard_callback(self, key):
               
        if key == 'm': # 按下m,更新 observation 中的 obj_pos
            if self.flag_object_key == 0: #判断按键是否和上一循环一样，避免重复触发
                self.flag_object = 1 #按键按下，状态置1。记忆本次循环按键状态
                
                print('flag_object:',self.flag_object)
                print('！！！！object:change！！！！')
                self.flag_object_key = 1 #按键按下，状态置1。记忆本次循环按键状态
        else:
            self.flag_object_key = 0 #按键松开，状态归零

    def Timer_Callback(self,event):

        if self.flag_object == 1 :
            print("self.obj_pos",self.obj_pos)
            self.updated_obj_pos = copy.deepcopy(self.palm_pos)
            self.updated_obj_pos[2] = self.updated_obj_pos[2] - 0.02  # 减去 0.02
            # 更新 observation 中的 obj_pos
            print("更新 observation 中的 obj_pos",self.updated_obj_pos)
            self.observation = np.concatenate([self.qpos_6[:-6], self.updated_obj_pos, self.palm_pos, self.target_pos])
        self.observation = np.concatenate([self.ur_pose,self.hand_joint_12,self.obj_pos,self.palm_pos,self.target_pos])
        self.action_12 = np.concatenate([self.ur_pose,self.hand_joint_12],axis=None)
        self.inspire_force = self.inspire_data[-6:]

      
    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        joint_freq = 1 / dt_helper(self.joint_timestamps)
        arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)
        gripper_command_freq = 1 / dt_helper(self.gripper_command_timestamps)

        print(f'{joint_freq=:.2f}\n{arm_command_freq=:.2f}\n{gripper_command_freq=:.2f}\n')


# def move_arms(L_hs_joint):
#     rospy.init_node('move_arms', anonymous=True)
#     publisher_ur = rospy.Publisher('/URSub', URSub, queue_size = 10)
#     pub_ur3_pose = URSub()
#     L_hs_joint_rad = [math.radians(deg) for deg in L_hs_joint]  # 转换为弧度
#     pub_ur3_pose.Joint_signals[0] = L_hs_joint_rad[0]
#     pub_ur3_pose.Joint_signals[1] = L_hs_joint_rad[1]
#     pub_ur3_pose.Joint_signals[2] = L_hs_joint_rad[2]
#     pub_ur3_pose.Joint_signals[3] = L_hs_joint_rad[3]
#     pub_ur3_pose.Joint_signals[4] = L_hs_joint_rad[4]
#     pub_ur3_pose.Joint_signals[5] = L_hs_joint_rad[5]
#     pub_ur3_pose.ControlMode = 2
#     publisher_ur.publish(pub_ur3_pose)

def move_hands(hans_joint):
    rospy.init_node('move_hands', anonymous=True)
    publisher2inspire = rospy.Publisher('dexmo_hand_joint', Int32MultiArray, queue_size=10)
    msg2inspire = Int32MultiArray()
    msg2inspire.data = hans_joint
    publisher2inspire.publish(msg2inspire)#转化为rosmsg发出去

