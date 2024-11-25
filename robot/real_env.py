import time
import numpy as np
import collections
import matplotlib.pyplot as plt
import dm_env
from pyquaternion import Quaternion
import rospy

# from robot.robot_utils import Recorder, ImageRecorder
# from robot.robot_utils import move_arms, move_hands
# from robot.robot_utils_hand import move_arms, move_hands
from robot.robot_utils_hand import move_hands
from robot.robot_utils_hand import Recorder, ImageRecorder
# from ur_msgs.msg import URSub
# from ur_msgs.msg import URPub_pose
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float32MultiArray,MultiArrayLayout,MultiArrayDimension
from scipy.spatial.transform import Rotation as R
from collections import deque
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import IPython
e = IPython.embed

    
class RealEnv_hand:
    """
    Environment for real robot manipulation
    Action space:      [hand_positions (6)    
    Observation space: {"qpos": 
                        Concat[ 
                                hand_position (6)           
                                hand_force (6)]    
                        
                        "images": {"cam_left": (480x640x3),  # h, w, c
    """

    def __init__(self, init_node, setup_robots=True, setup_base=False,num_stack=3):
        self.num_stack = num_stack
        rospy.init_node('real_robot_hand', anonymous=True)

        self.recorder = Recorder(init_node=False,num_stack=self.num_stack)
        self.image_recorder = ImageRecorder(init_node=False)

        self.publisher2inspire = rospy.Publisher('dexmo_hand_joint', Int32MultiArray, queue_size=10)    
        self.publisher2weight = rospy.Publisher('weight', Int32MultiArray, queue_size=10)  
        self.publisher2scores= rospy.Publisher('scores', Float32MultiArray, queue_size=10)  
        self.publisher2force = rospy.Publisher('pre_force', Int32MultiArray, queue_size=10)  

    def _reset_joints(self):
        self.L_hs_joint = [0, -98, -62, -211, -90, 180]

    def _reset_hands(self):
        self.hans_joint = [1000, 1000, 1000, 1000, 1000, 500]
        move_hands(self.hans_joint)

    def get_observation(self, get_tracer_vel=False):
        obs = collections.OrderedDict()
        obs['joint'] =  self.recorder.joints
        obs['force'] =  self.recorder.forces
        obs['image'] = self.image_recorder.get_images(self.num_stack)
        return obs

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        if not fake:
            self._reset_hands()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(self.num_stack))
    

    def step(self, args,action, force,base_action=None, get_tracer_vel=False, get_obs=True):

        msg2force = Int32MultiArray()
        msg2force.data = [
            int(force[0] * 2000-1000), 
            int(force[1] * 2000-1000), 
            int(force[2] * 2000-1000),
            int(force[3] * 2000-1000), 
            int(force[4] * 2000-1000), 
            int(force[5] * 2000-1000)
        ]
        self.publisher2force.publish(msg2force)


        msg2inspire = Int32MultiArray()
        msg2inspire.data = [
            int(action[0][0] * args.inspire), 
            int(action[0][1] * args.inspire), 
            int(action[0][2] * args.inspire),
            int(action[0][3] * args.inspire), 
            int(action[0][4] * args.inspire), 
            int(action[0][5] * args.inspire)
        ]
        self.publisher2inspire.publish(msg2inspire)

        if get_obs:
            obs = self.get_observation(self.num_stack)
        else:
            obs = None

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs)
    
def make_real_env_hand(init_node, setup_robots=True, setup_base=False,num_stack=3):
    num_stack = num_stack
    env = RealEnv_hand(init_node, setup_robots, setup_base,num_stack)
    return env



if __name__ == '__main__':
    make_real_env_hand()


