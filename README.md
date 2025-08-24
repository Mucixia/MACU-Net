# MACU-Net
The repository contains the implementation of MACUNet, a deep learning-based framework for sparse-view CT reconstruction. MACUNet is a memory-augmented unfolding network that incorporates a Side-Information (SI) mechanism and a Cross-stage Memory-Enhancement Module (CMEM) to improve inter-stage communication and capture long-range dependencies.
## Installation
1. Create a Virtual Environment:
'python3 -m venv venv'
2. Install Dependencies:
'pip install -r requirements.txt'
3. Install PyTorch 1.7.1:
'pip install torch==1.7.1'
4. Install Torch-radon: The projection transformation library used in this project is Torch-radon. To begin using it, please follow the installation instructions provided in the [Torch-radon](https://github.com/matteo-ronchetti/torch-radon).
## Usgae
1. Dataset: We evaluate on the''Low Dose CT Image and Projection Data'' [dataset](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/).
2. Training: run 'python CT_MACU.py'; set --run_mode='train'.
3. Testing: run 'python CT_MACU.py'; set --run_mode='test'.
## Example evaluation
The testing weights for the model are located at: './model_and_result/CT-fan-MACU-ds64_layer_8_lr_0.000100/net_params_50.pkl'.
## Acknowledgment
We would like to thank [ISTA-Net](https://github.com/jianzhangcs/ISTA-Net/tree/master) and [MVMS-RCN](https://github.com/fanxiaohong/MVMS-RCN) for providing the open-source code that inspired or supported this project.

