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

# class RealEnv_hand_arm:
#     """
#     Environment for real robot manipulation
#     Action space:      [arm_qpos (6)                  # absolute tcp position
#                         hand_positions (6)    
#     Observation space: {"qpos": 
#                         Concat[ arm_qpos (6)          # absolute tcp position
#                                 hand_position (6)    
#                                 object_position (3)        
#                                 hand_force (6)]    
                        
#                         "images": {"cam_left": (480x640x3),  # h, w, c
#     """

#     def __init__(self, init_node, setup_robots=True, setup_base=False):

#         rospy.init_node('real_robot', anonymous=True)
#         ######### pid #########
#         self.pos_err_relax = 0
#         self.pos_err = [0.0, 0.0, 0.0]
#         self.pos_err_integral = [0.0, 0.0, 0.0]
#         self.pos_err_last = [0.0, 0.0, 0.0]
#         self.pos_speed_max = 0
#         self.pos_speed_min = 0
        
#         self.rv_err_relax = 0
#         self.rv_err = [0.0, 0.0, 0.0]
#         self.rv_err_integral = [0.0, 0.0, 0.0]
#         self.rv_err_last = [0.0, 0.0, 0.0]
#         self.rv_speed_max = 0
#         self.rv_speed_min = 0

#         self.pid_pos_Kp = 0
#         self.pid_pos_Ki = 0
#         self.pid_pos_Kd = 0
#         self.pid_rv_Kp = 0
#         self.pid_rv_Ki = 0
#         self.pid_rv_Kd = 0

#         self.current_velocity = [0, 0, 0]
#         self.current_velocity_r = [0, 0, 0]


#         if rospy.has_param('data_ur_pub_name'):
#             data_ur_pub_name=rospy.get_param('data_ur_pub_name',default='/URSub')
#             print('data_ur_pub_name:',data_ur_pub_name)
#         else:
#             print('parameter data_ur_pub_name not found!!!')

#         if rospy.has_param('data_inspire_pub_name'):
#             data_inspire_pub_name=rospy.get_param('data_inspire_pub_name',default='dexmo_hand_joint')
#             print('data_inspire_pub_name:',data_inspire_pub_name)
#         else:
#             print('parameter data_inspire_pub_name not found!!!')

#         if rospy.has_param('ur_sub_name'):
#             ur_sub_name = rospy.get_param('ur_sub_name',default = '/UR_pose_Pub')
#             print('ur_sub_name:',ur_sub_name)
#         else:
#             print('parameter ur_sub_name not found!!!')

#         ##### PID  参数  
#         if rospy.has_param('pid_pos_Kp'):
#             self.pid_pos_Kp = rospy.get_param('pid_pos_Kp',default = 10)
#             print('pid_pos_Kp:',self.pid_pos_Kp)
#         else:
#             print('parameter pid_pos_Kp not found!!!')

#         if rospy.has_param('pid_pos_Ki'):
#             self.pid_pos_Ki = rospy.get_param('pid_pos_Ki',default = 0)
#             print('pid_pos_Ki:',self.pid_pos_Ki)
#         else:
#             print('parameter pid_pos_Ki not found!!!')

#         if rospy.has_param('pid_pos_Kd'):
#             self.pid_pos_Kd = rospy.get_param('pid_pos_Kd',default = 0)
#             print('pid_pos_Kd:',self.pid_pos_Kd)
#         else:
#             print('parameter pid_pos_Kd not found!!!')

#         if rospy.has_param('pid_rv_Kp'):
#             self.pid_rv_Kp = rospy.get_param('pid_rv_Kp',default = 10)
#             print('pid_rv_Kp:',self.pid_rv_Kp)
#         else:
#             print('parameter pid_rv_Kp not found!!!')

#         if rospy.has_param('pid_rv_Ki'):
#             self.pid_rv_Ki = rospy.get_param('pid_rv_Ki',default = 0)
#             print('pid_rv_Ki:',self.pid_rv_Ki)
#         else:
#             print('parameter pid_rv_Ki not found!!!')

#         if rospy.has_param('pid_rv_Kd'):
#             self.pid_rv_Kd = rospy.get_param('pid_rv_Kd',default = 0)
#             print('pid_rv_Kd:',self.pid_rv_Kd)
#         else:
#             print('parameter pid_rv_Kd not found!!!')

#         if rospy.has_param('pos_err_relax'):
#             self.pos_err_relax = rospy.get_param('pos_err_relax',default = 0.003)
#             print('pos_err_relax:',self.pos_err_relax)
#         else:
#             print('parameter pos_err_relax not found!!!')

#         if rospy.has_param('pos_speed_max'):
#             self.pos_speed_max = rospy.get_param('pos_speed_max',default = 1.0)
#             print('pos_speed_max:',self.pos_speed_max)
#         else:
#             print('parameter pos_speed_max not found!!!')

#         if rospy.has_param('pos_speed_min'):
#             self.pos_speed_min = rospy.get_param('pos_speed_min',default = 3.14)
#             print('pos_speed_min:',self.pos_speed_min)
#         else:
#             print('parameter pos_speed_min not found!!!')

#         if rospy.has_param('rv_err_relax'):
#             self.rv_err_relax = rospy.get_param('rv_err_relax',default = 0.01)
#             print('rv_err_relax:',self.rv_err_relax)
#         else:
#             print('parameter rv_err_relax not found!!!')

#         if rospy.has_param('rv_speed_max'):
#             self.rv_speed_max = rospy.get_param('rv_speed_max',default = 3.14)
#             print('rv_speed_max:',self.rv_speed_max)
#         else:
#             print('parameter rv_speed_max not found!!!')

#         if rospy.has_param('rv_speed_min'):
#             self.rv_speed_min = rospy.get_param('rv_speed_min',default = 3.14)
#             print('rv_speed_min:',self.rv_speed_min)
#         else:
#             print('parameter rv_speed_min not found!!!')
#         self.recorder = Recorder(init_node=False)
#         self.image_recorder = ImageRecorder(init_node=False)

#         self.publisher2inspire = rospy.Publisher(data_inspire_pub_name, Int32MultiArray, queue_size=10)    
#         self.publisher_ur = rospy.Publisher(data_ur_pub_name, URSub, queue_size = 10) 
#         ur_sub = rospy.Subscriber(ur_sub_name,URPub_pose,self.ur_callback) 
    
#     def _reset_joints(self):
#         self.L_hs_joint = [0, -98, -62, -211, -90, 180]
#         move_arms(self.L_hs_joint)

#     def _reset_hands(self):
#         self.hans_joint = [1000, 1000, 1000, 1000, 1000, 1000]
#         move_hands(self.hans_joint)

#     def get_observation(self, get_tracer_vel=False):
#         obs = collections.OrderedDict()
#         obs['qpos'] =  self.recorder.observation
#         obs['images'] = self.image_recorder.get_images()
  
#         return obs

#     def get_reward(self):
#         return 0

#     def reset(self, fake=False):
#         if not fake:
#             self._reset_joints()
#             self._reset_hands()
#         return dm_env.TimeStep(
#             step_type=dm_env.StepType.FIRST,
#             reward=self.get_reward(),
#             discount=None,
#             observation=self.get_observation())
    
#     def ur_callback(self,TCP_Position):
#         self.ur_pose[0] = TCP_Position.TCP_Position[0]
#         self.ur_pose[1] = TCP_Position.TCP_Position[1]
#         self.ur_pose[2] = TCP_Position.TCP_Position[2]
#         self.ur_pose[3] = TCP_Position.TCP_Position[3]
#         self.ur_pose[4] = TCP_Position.TCP_Position[4]
#         self.ur_pose[5] = TCP_Position.TCP_Position[5]
#         # print('self.ur_pose',self.ur_pose)


#     def step(self, action, base_action=None, get_tracer_vel=False, get_obs=True):
#         state_len = int(len(action) / 2)
#         hands_action = action[0:6]
#         arm_action = action[6:]
#         pub_ur3_pose = URSub()
#         pub_ur3_pose.ControlMode = 3 # (TCP_SPEED_MODE)
#         pub_ur3_pose.Pose_signals[0:3] = action[0:3]
#         pub_ur3_pose.Pose_signals[3:6] = action[3:6]
#         self.publisher_ur.publish(pub_ur3_pose)
#         msg2inspire = Int32MultiArray()
#         pub_ur3_pose = URSub()
#         msg2inspire.data = [int(action[6]*1000), int(action[8]*1000), int(action[10]*1000),
#                             int(action[12]*1000), int(action[14]*1000), int(action[16]*1000)]
#         # print('msg2inspire.data:', msg2inspire.data)
#         self.publisher2inspire.publish(msg2inspire)
#         if get_obs:
#             obs = self.get_observation(get_tracer_vel)
#         else:
#             obs = None

#         return dm_env.TimeStep(
#             step_type=dm_env.StepType.MID,
#             reward=self.get_reward(),
#             discount=None,
#             observation=obs)
    
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

    def __init__(self, init_node, setup_robots=True, setup_base=False):
        rospy.init_node('real_robot_hand', anonymous=True)
        # if rospy.has_param('data_inspire_pub_name'):
        #     data_inspire_pub_name=rospy.get_param('data_inspire_pub_name',default='dexmo_hand_joint')
        #     print('data_inspire_pub_name:',data_inspire_pub_name)
        # else:
        #     print('parameter data_inspire_pub_name not found!!!')
        self.recorder = Recorder(init_node=False)
        self.image_recorder = ImageRecorder(init_node=False)

        self.publisher2inspire = rospy.Publisher('dexmo_hand_joint', Int32MultiArray, queue_size=10)    
        self.publisher2weight = rospy.Publisher('weight', Int32MultiArray, queue_size=10)  
        self.publisher2scores= rospy.Publisher('scores', Float32MultiArray, queue_size=10)  
        self.publisher2force = rospy.Publisher('pre_force', Int32MultiArray, queue_size=10)  

    def _reset_joints(self):
        self.L_hs_joint = [0, -98, -62, -211, -90, 180]
        # move_arms(self.L_hs_joint)

    def _reset_hands(self):
        self.hans_joint = [1000, 1000, 1000, 1000, 1000, 500]
        move_hands(self.hans_joint)

    def get_observation(self, get_tracer_vel=False):
        obs = collections.OrderedDict()
        obs['joint'] =  self.recorder.joints
        obs['force'] =  self.recorder.forces
        obs['image'] = self.image_recorder.get_images()
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
            observation=self.get_observation())
    
    # def step(self, action, weight,base_action=None, get_tracer_vel=False, get_obs=True):
    def step(self, args,scores,action, force,base_action=None, get_tracer_vel=False, get_obs=True):
        msg_weight = Int32MultiArray()
        # msg_force = Float32MultiArray()

        msg2force = Int32MultiArray()
        # print('force',force)
        # print('action',action)
        msg2force.data = [
            int(force[0] * 2000-1000), 
            int(force[1] * 2000-1000), 
            int(force[2] * 2000-1000),
            int(force[3] * 2000-1000), 
            int(force[4] * 2000-1000), 
            int(force[5] * 2000-1000)
        ]
        print('force.data:', msg2force.data)
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
        print('inspire.data:', msg2inspire.data)
        self.publisher2inspire.publish(msg2inspire)
        # msg2inspire.data = [int(action[0][0]), int(action[0][1]), int(action[0][2]), int(action[0][3]), int(action[0][4]), int(action[0][5])]
        # msg_scores = Float32MultiArray()
        # msg_scores.data = scores
        # # print('scores.data:', msg_scores.data)
        # self.publisher2scores.publish(msg_scores)

        # weight_int = np.round(weight * args.inspire).astype(np.int32)
        # weight_float = weight.astype(np.float32) 
        # print('weight_float',weight_float)
        # print('weight_int',weight_int)

        # # 创建 MultiArrayLayout
        # layout = MultiArrayLayout()
        # layout.dim = [MultiArrayDimension()]
        # layout.dim[0].label = "weights"
        # layout.dim[0].size = weight_int.shape[1]
        # layout.dim[0].stride = weight_int.shape[1] * weight_int.shape[0]
        # msg_weight.layout = layout
        # msg_weight.data = weight_int.flatten().tolist() 
        # self.publisher2weight.publish(msg_weight)

        # msg_force.data = msg_force
        # self.publisher2force.publish(msg_force)

        if get_obs:
            obs = self.get_observation(get_tracer_vel)
        else:
            obs = None

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs)
    
def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14) # 6 joint + 1 gripper, for two arms
    # Arm actions
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # Gripper actions
    # action[6] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.dxl.joint_states.position[6])
    # action[7+6] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])

    return action


# def make_real_env_hand_arm(init_node, setup_robots=True, setup_base=False):
#     env = RealEnv_hand_arm(init_node, setup_robots, setup_base)
#     return env


def make_real_env_hand(init_node, setup_robots=True, setup_base=False):
    env = RealEnv_hand(init_node, setup_robots, setup_base)
    return env

def test_real_teleop():
    """
    Test bimanual teleoperation and show image observations onscreen.
    It first reads joint poses from both master arms.
    Then use it as actions to step the environment.
    The environment returns full observations including images.

    An alternative approach is to have separate scripts for teleoperation and observation recording.
    This script will result in higher fidelity (obs, action) pairs
    """

    onscreen_render = True
    render_cam = 'cam_left_wrist'


    # setup the environment
    env = make_real_env(init_node=False)
    ts = env.reset(fake=True)
    episode = [ts]
    # setup visualization
    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam])
        plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        # if onscreen_render:
        #     plt_img.set_data(ts.observation['images'][render_cam])
        #     plt.pause(DT)
        # else:
        #     time.sleep(DT)


if __name__ == '__main__':
    test_real_teleop()

