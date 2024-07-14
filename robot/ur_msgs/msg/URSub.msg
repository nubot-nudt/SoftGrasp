Header header

#关节和位姿
float64[6] Joint_signals
float64[6] Pose_signals
float64[6] TCP_Speed

# 1 代表关节 2代表末端
# JOINT_SPEED_MODE = 1
# JOINT_POS_MODE = 2
# TCP_SPEED_MODE = 3
# TCP_POS_MODE = 4
int32 ControlMode