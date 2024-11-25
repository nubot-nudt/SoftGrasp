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
        self.cv_img_buffer = []  
        if init_node:
            rospy.init_node('image_recorder', anonymous=True)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_Callback)
    
        time.sleep(0.5)

    def image_Callback(self,data):
        self.cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.timestr = "%.6f" %  data.header.stamp.to_sec()   
        self.cv_img_buffer.append((self.timestr, self.cv_img))

    def get_images(self,num_stack):
        stacked_images = []

        for i in range(num_stack):
            if i < len(self.cv_img_buffer):
                stacked_images.append(self.cv_img_buffer[i][1])
            else:
                stacked_images.append(None)  

        return stacked_images

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
    def __init__(self,init_node=True, is_debug=False,num_stack=3):
        from collections import deque
        import rospy
        from sensor_msgs.msg import JointState
        self.num_stack = num_stack

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


        if init_node:
            rospy.init_node('recorder', anonymous=True)

        inspire_sub = rospy.Subscriber('inspire_angle_force', Int32MultiArray, self.inspire2dexmo_Callback)


    
        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
            self.gripper_command_timestamps = deque(maxlen=50)
        time.sleep(0.1)

   

    def inspire2dexmo_Callback(self, inspire_msg):
        self.inspire_data = inspire_msg.data
        self.inspire_data = np.array(self.inspire_data)
        
        if len(self.joint) > self.num_stack:
            self.joint = self.joint[-self.num_stack:]

        self.joints = np.vstack(self.joint)

        if not hasattr(self, 'torque'):
            self.torque = []

        self.torque.append(self.inspire_data[6:12])

        if len(self.torque) > self.num_stack:
            self.torque = self.torque[-self.num_stack:]

        self.torques = np.vstack(self.torque)
     

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        joint_freq = 1 / dt_helper(self.joint_timestamps)
        arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)
        gripper_command_freq = 1 / dt_helper(self.gripper_command_timestamps)

        print(f'{joint_freq=:.2f}\n{arm_command_freq=:.2f}\n{gripper_command_freq=:.2f}\n')


def move_hands(hans_joint):
    publisher2inspire = rospy.Publisher('dexmo_hand_joint', Int32MultiArray, queue_size=10)
    msg2inspire = Int32MultiArray()
    msg2inspire.data = hans_joint
    publisher2inspire.publish(msg2inspire)

