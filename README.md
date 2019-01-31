Fractures detection on X-Ray images using Machine Learning
# Preparation
1. [Keras](https://keras.io/)
1. [MobileNetV2](https://arxiv.org/abs/1801.04381)
1. [MURA](https://arxiv.org/abs/1712.06957)
# Performance
1. Model: MobileNetV2
1. Dataset: XR_HUMERUS
1. Accuracy: 68.9%

![Aaron Swartz](https://raw.githubusercontent.com/Arith2/yuzhu_MURA_MobileNet/master/steps100_acc_loss_draw1.jpg)
# Log
```json
{
    "Model": "MobileNetV2",
    "Class": "XR_HUMERUS",
    "Time_for_training": "2:11:51.668540",
    "Loss": [
        0.7514553318483682,
        0.6548915859677507,
        0.614088291440906,
        0.6020854084909981,
        0.5844772238223279,
        0.5891418776759775,
        0.556057234279445,
        0.5627546866458001
    ],
    "Accuracy": [
        0.5680958386252236,
        0.6380620646355951,
        0.6834804542353294,
        0.6912602916633577,
        0.7052332912237016,
        0.6925269158072227,
        0.7244640607927065,
        0.72070157097651737
    ],
    "Loss_valid": [
        2.4282438686915806,
        5.077249728308784,
        3.416385951874748,
        2.47827467388576157,
        0.6141623440242949,
        0.87005202107951572,
        0.8192852735519409,
        0.79446715702553976
    ],
    "Acc_valid": [
        0.46349206377589514,
        0.4952380955219269,
        0.5015873019657437,
        0.5523809514348469,
        0.666666665152898,
        0.5682539686324104,
        0.5873015863554818,
        0.6258064510360841
    ],
    "Layers": 157,
    "Parameters": 2259970,
    "Steps": 100
}
```
