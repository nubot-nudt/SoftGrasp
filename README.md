# SoftGrasp: Adaptive Grasping for Dexterous Hand based on Multimodal Fusion Imitation Learning

### Project Page | Video |Arxiv
This repo contains the implementation of our paper:
> **SoftGrasp: Adaptive Grasping for Dexterous Hand based on Multimodal Fusion Imitation Learning**
> 
> [YiHong Li](https://github.com/swagyiyi),[Ce Guo](https://github.com/henghenghahei849),[JunKai Ren](https://github.com/jkren6),HuiZhang ,HuiMin Lu
>

**The code  will be released after our paper  is accepted.**

## install
run:
```
conda create -n "SoftGrasp" python=3.7 -y && conda activate multimodal
pip install -r requirements.txt
```

To train policy with mutil_model
```
python SoftGrasp_train.py
```
train_utils.py ：

 trainer.fit(
        pl_module,
        ckpt_path='exp05132024j_f_vf/imi_learn_1/checkpoints/j_f_vf_share/last.ckpt'
        # ckpt_path=None
        if args.resume is None
        else os.path.join(os.getcwd(), args.resume),
    )

ckpt_path （Resume training from previous policy）
```
python visualize_real.py 

python visualize_policy.py 
```
episode_times_1.csv
| file      | Description |
| ----------- | ----------- |
| train.csv                 | train_dataset                            |
| val.csv                   | Val_dataset                               |
| val.csv                   | test_dataset                               |
| exp_apple_1           | Fixed camera captures images                           |
| exp_apple_1.pickle    | Contains human demonstration actions        |


Here are what each symbol means:

| Symbol      | Description |
| ----------- | ----------- |
| I   | camera input from a fixed perspective        |
| A   | joint angle of dexterous hands |
| T   | Joint torque of dexterous hands |


### Evaluate your results
To view your model's results, run <br>
 ```conda activate SoftGrasp```
 
```tensorboard --logdir exp{data}{task}```

| Description   |     pmulsa      |
| -----------   |   -----------   | 
|DATA           | SoftGrasp_dataset.py  |
|Dataset        | base.py       |
|ImiEngine      | engine.py     |
|imitation_model| SoftGrasp_models.py |

Physical experiment：
```
cd Data/code/hand/two_hand
source devel/setup.bash
roslaunch inspire_hand inspire_hand_control.launch
```
```
cd Data/reslsense-d435i/
source devel/setup.bash
roslaunch realsense2_camera demo_pointcloud.launch

<!-- roslaunch realsense2_camera rs_camera.launch -->

cd Data/code/hand/two_hand
source devel/setup.bash
roslaunch handeye-calib aruco_start_realsense_sdk.launch

```

 ```conda activate SoftGrasp```

```python visualize_real.py ```

```
source devel/setup.bash

roslaunch inspire_hand hand_control.launch test_flag:=1

source devel/setup.bash

rosservice call /inspire_hand/set_angle 1000 1000 1000 1000 1000 1000

rosservice call /inspire_hand/set_clear_error

rosservice call /inspire_hand/set_force_clb
```
