# SoftGrasp Dexterous Hand Manipulation


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

ckpt_path 从之前的模型恢复训练
```
python visualize_real.py 

python visualize_policy.py 
```
episode_times_1.csv
| file      | Description |
| ----------- | ----------- |
| train.csv                 | 训练的的数据集                             |
| val.csv                   | 测试的数据集                               |
| test_recordings             | 实验数据集，内部命名按照episode_times_1的内容 |
| exp_apple_1           | 固定相机拍摄画面                           |
| exp_apple_1.pickle    | 包含人类演示动作action和observation        |


Here are what each symbol means:

| Symbol      | Description |
| ----------- | ----------- |
| I   | camera input from a fixed perspective        |
| A   | The end and finger joints of a robotic arm |机械臂的末端和手指关节
| T   | Joint torque of dexterous hands |灵巧手的关节力矩


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

实物实验：
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