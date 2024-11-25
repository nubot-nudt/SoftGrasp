# SoftGrasp: Adaptive Grasping for Dexterous Hand based on Multimodal Fusion Imitation Learning

### Project Page | Video |Arxiv
This repo contains the implementation of our paper:
> **SoftGrasp: Adaptive Grasping for Dexterous Hand based on Multimodal Fusion Imitation Learning**
> 
> [YiHong Li](https://github.com/swagyiyi),[Ce Guo](https://github.com/henghenghahei849),[JunKai Ren](https://github.com/jkren6),[Bailiang Chen](https://github.com/skywalker1941),HuiZhang ,HuiMin Lu
>




## install

```
conda create -n "SoftGrasp" python=3.7 -y && conda activate multimodal
pip install -r requirements.txt
```

To train policy with mutil_model
```
python SoftGrasp_train.py
```
Test mutil_model
```
python visualize_real.py 
```
episode_times_1.csv
| file      | Description |
| ----------- | ----------- |
| train.csv                 | 	train_dataset                             |
| val.csv                   | 	Val_dataset                           |
| test_recordings             | dataset|
| exp_apple_21           | Fixed camera captures images                       |
| exp_apple_21.pickle    | Contains human demonstration actions      |


Here are what each symbol means:

| Symbol      | Description |
| ----------- | ----------- |
| I   | camera input from a fixed perspective        |
| A   | The end and finger joints of a robotic arm |
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


