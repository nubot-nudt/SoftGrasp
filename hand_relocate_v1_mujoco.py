from gym import error,spaces,utils
from gym.utils import seeding
import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
import mujoco_py
import os
import pickle
import numpy as np
# from hand_imitation.env.utils.mjcf_utils import xml_path_completion, array_to_string, find_elements
# from hand_imitation.env.models.base import MujocoXML
from robohive.envs.obs_vec_dict import ObsVecDict
# from hand_imitation.env.models.base import MujocoModel
import collections
import xml.etree.ElementTree as ET
# from filterpy.kalman import KalmanFilter
import copy
import cv2
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
ADD_BONUS_REWARDS = True
def make_sim_env():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    physics = mujoco.Physics.from_xml_string(open('/home/nubot-11/Data/hand_dapg-master/dapg/mj_envs-master/mj_envs/hand_manipulation_suite/assets/hand_relocate_1.xml').read())
    env = control.Environment(physics,HandEnvV1(random=False),time_limit=20, control_timestep=0.02,n_sub_steps=None, flat_observation=False)
    return env

class HandEnvV1(base.Task):
    def __init__(self, random=None):
        super().__init__(random=None)
       
    def get_reward(self, physics):
        site_pos = physics.data.site_xpos.copy()
        body_pos = physics.data.xpos.copy()
        obj_pos = body_pos[16, :]
        target_pos = site_pos[33, :]
        palm_pos = site_pos[0:1]
        
        reward = -0.1*np.linalg.norm(palm_pos-obj_pos)       
        if np.linalg.norm(palm_pos-obj_pos) < 0.02:
            reward += 0.1  
            if obj_pos[2] > 0.04:            
                reward += 1.0 
                reward += -0.5*np.linalg.norm(palm_pos-target_pos)      
                reward += -0.5*np.linalg.norm(obj_pos-target_pos)       

        if ADD_BONUS_REWARDS:
            if np.linalg.norm(obj_pos-target_pos) < 0.1:
                reward += 10.0                                          
            if np.linalg.norm(obj_pos-target_pos) < 0.05:
                reward += 20.0                                         
    
        goal_achieved = True if np.linalg.norm(obj_pos-target_pos) < 0.1 else False
        return reward
    def before_step(self, action, physics):
        action = np.clip(action, -1.0, 1.0)
        try:
            env_action = self.act_mid + action*self.act_rng 
        except:
            env_action = action 
        # print('env_action',env_action.shape)
        super().before_step(env_action, physics)
        return
    def step(self, a, physics):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng 
        except:
            a = a  

        # self.do_simulation(a, self.frame_skip)

        ob = self.get_observation(physics)
        # try:
        #     image = self.viewer2.render(height=640, width=480, camera_name='fixed', depth=False)
        #     physics.forward()
        #     self.skip_temp += 1
        #     if self.skip_temp == 10:
        #         image = image[...,::-1]
        #         image = cv2.flip(image, 0)
        #         cv2.imwrite(self.curr_dir+'/pictures/'+str(self.pic_index)+'.jpg', image)
        #         self.skip_temp = 0
        # except:
        #     self.viewer2 = mujoco.MjRenderContextOffscreen(physics, 0)
        site_pos = physics.data.site_xpos.copy()
        body_pos = physics.data.xpos.copy()
        obj_pos = body_pos[16, :]
        target_pos = site_pos[33, :]
        palm_pos = site_pos[0:1]
    

        reward = -0.1*np.linalg.norm(palm_pos-obj_pos)       
        if np.linalg.norm(palm_pos-obj_pos) < 0.02:
            reward += 0.1  
            if obj_pos[2] > 0.04:            
                reward += 1.0 
                reward += -0.5*np.linalg.norm(palm_pos-target_pos)      
                reward += -0.5*np.linalg.norm(obj_pos-target_pos)       

        if ADD_BONUS_REWARDS:
            if np.linalg.norm(obj_pos-target_pos) < 0.1:
                reward += 10.0                                          
            if np.linalg.norm(obj_pos-target_pos) < 0.05:
                reward += 20.0                                         
    
        goal_achieved = True if np.linalg.norm(obj_pos-target_pos) < 0.1 else False

        return ob, reward, False, dict(goal_achieved=goal_achieved)
    
    def get_observation(self,physics):
        obs = collections.OrderedDict()
        qpos_raw = physics.data.qpos.copy()
        qvel_raw = physics.data.qvel.copy()
        site_pos = physics.data.site_xpos.copy()
        body_pos = physics.data.xpos.copy()
        obj_pos = body_pos[16, :]
        target_pos = site_pos[33, :]
        palm_pos = site_pos[0:1]
        obs['qpos'] = qpos_raw[0:18]
        obs['qvel'] = qvel_raw[0:18]
        obs['obj_pos']  = obj_pos
        obs['target_pos'] = target_pos
        obs['palm_pos'] = palm_pos
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        # 将二维数组转换为一维数组
        palm_pos = np.squeeze(palm_pos)
        obj_pos = np.squeeze(obj_pos)
        target_pos = np.squeeze(target_pos)
        state = np.concatenate([qpos_raw[:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos])
        observation = np.concatenate([qpos_raw[:-6], obj_pos, palm_pos, target_pos])
        obs['state'] = state
        obs['observation'] = observation
        return obs
      
    @staticmethod
    def get_env_state(self, physics):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = physics.data.qpos.ravel().copy()
        qv = physics.data.qvel.ravel().copy()
        hand_qpos = qp[:30]
        obj_pos = physics.data.body_xpos[self.obj_bid].ravel()
        palm_pos = physics.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = physics.data.site_xpos[self.target_obj_sid].ravel()
        return dict(hand_qpos=hand_qpos, obj_pos=obj_pos, target_pos=target_pos, palm_pos=palm_pos,
                    qpos=qp, qvel=qv)

    # def set_env_state(self, state_dict, physics):
    #     """
    #     Set the state which includes hand as well as objects and targets in the scene
    #     """
    #     qp = state_dict['qpos']
    #     qv = state_dict['qvel']
    #     print("qp.shape:", qp.shape)
    #     print("qv.shape:", qv.shape)
    #     obj_pos = state_dict['obj_pos']
    #     target_pos = state_dict['target_pos']
    #     physics.data.qpos[:] = qp
    #     physics.data.qvel[:] = qv
    #     physics.model.body_pos[self.obj_bid] = obj_pos
    #     physics.model.site_pos[self.target_obj_sid] = target_pos
    #     physics.forward()

    def initialize_episode(self, physics):
        """设置每个 episode 开始时环境的状态。"""
        # TODO 注意：该函数不随机化环境配置。相反，从外部设置 BOX_POSE
        # 重置 qpos、控制和盒子位置
        START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239, 0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239, 0, 0]
        BOX_POSE = [0,0,0 ]
        with physics.reset_context():
            physics.named.data.qpos[:18] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

 
    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success * 100.0 / num_paths
        return success_percentage
