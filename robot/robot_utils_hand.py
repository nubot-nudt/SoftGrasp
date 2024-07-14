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
        self.cv_img_buffer = []  # 用于存储多个图像数据的缓冲区
        if init_node:
            rospy.init_node('image_recorder', anonymous=True)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_Callback)
    
        time.sleep(0.5)

    def image_Callback(self,data):
        self.cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.timestr = "%.6f" %  data.header.stamp.to_sec()   
        self.cv_img_buffer.append((self.timestr, self.cv_img))

    def get_images(self):
        stacked_images = []
        for i in range(3):
            if i < len(self.cv_img_buffer):
                stacked_images.append(self.cv_img_buffer[i][1])
            else:
                stacked_images.append(None)  # 如果某个时间步长没有数据，可以添加 None
        # print(stacked_images)

        return stacked_images

        # image_dict = dict()
        # if hasattr(self, 'cv_img') and self.cv_img is not None:
        #     image_dict = getattr(self, f'cv_img')
        # return image_dict

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
        self.joints = None
        self.forces = None


        self.is_debug = is_debug
        # self.settings = self.saveTerminalSettings()

        if init_node:
            rospy.init_node('recorder', anonymous=True)

        inspire_sub = rospy.Subscriber('inspire_angle_force', Int32MultiArray, self.inspire2dexmo_Callback)

        # dexmo_sub = rospy.Subscriber('/dexmo_hand_joint', Int32MultiArray, self.dexmo_Callback)
        # data_sample_Timer = rospy.Timer(rospy.Duration(1.0 / 10.0), self.Timer_Callback) 
       

        # try:
        #     while(1):
        #         key = self.getKey(self.settings)
        #         self.keyboard_callback(key)
        #         if key == '\x03':
        #             break

        # except Exception as e:
        #     print(e)
    
        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
            self.gripper_command_timestamps = deque(maxlen=50)
        time.sleep(0.1)

    # def saveTerminalSettings(self):
    #     if sys.platform == 'win32':
    #         return None
    #     return termios.tcgetattr(sys.stdin)

    # def getKey(self,settings):
    #     if sys.platform == 'win32':
    #         # getwch() returns a string on Windows
    #         key = msvcrt.getwch()
    #     else:
    #         tty.setraw(sys.stdin.fileno())
    #         # sys.stdin.read() returns a string on Linux
    #         key = sys.stdin.read(1)
    #         termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    #     return key

    def inspire2dexmo_Callback(self, inspire_msg):
        self.inspire_data = inspire_msg.data
        self.inspire_data = np.array(self.inspire_data)
        # print('self.inspire_data',self.inspire_data) 
        # self.inspire_data = self.inspire_data /1000 
        # self.inspire_joint = self.inspire_data[0:6] 
        # self.observation = self.inspire_data
        # 检查是否已经有三个时间步长的数据，如果没有，则创建一个空列表
        if not hasattr(self, 'joint'):
            self.joint = []

        self.joint.append(self.inspire_data[0:6])

        if len(self.joint) > 3:
            self.joint = self.joint[-3:]

        self.joints = np.vstack(self.joint)

        if not hasattr(self, 'force'):
            self.force = []

        self.force.append(self.inspire_data[6:12])

        if len(self.force) > 3:
            self.force = self.force[-3:]

        self.forces = np.vstack(self.force)
     

    # def dexmo_Callback(self,data):
    #     i = 0
    #     for i in range(min(len(data.data), len(self.Arr))):
    #         self.Arr[i] = (data.data[i])*0.001
    #         self.dexmo =self.Arr

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
    # rospy.init_node('move_hands', anonymous=True)
    publisher2inspire = rospy.Publisher('dexmo_hand_joint', Int32MultiArray, queue_size=10)
    msg2inspire = Int32MultiArray()
    msg2inspire.data = hans_joint
    publisher2inspire.publish(msg2inspire)#转化为rosmsg发出去

