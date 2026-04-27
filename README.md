MIMO-SST (Multi-Input Multi-Output Spatial-Spectral Transformer) 

This project is an open-source code implementation for the task of Hyperspectral and Multispectral Image Fusion (HSI-MSI Fusion), corresponding to the paper "MIMO-SST: Multi-Input Multi-Output Spatial-Spectral Transformer for Hyperspectral and Multispectral Image Fusion" (IEEE TGRS 2024).

📌 Project Introduction

This repository provides the PyTorch implementation of the MIMO-SST network. This network is a novel Multi-Input Multi-Output Spatial-Spectral Transformer architecture designed to address the Hyperspectral Image (HSI) super-resolution problem, i.e., fusing a Low-Resolution Hyperspectral Image (LR-HSI) and a High-Resolution Multispectral Image (HR-MSI) to generate a High-Resolution Hyperspectral Image (HR-HSI) with high spatial and spectral resolution.

Core Innovations:
• Multi-Input Multi-Output (MIMO) Framework: Supervises the generation of HR-HSI at different scales through a coarse-to-fine architecture, thereby more comprehensively capturing image details and structures.
• Mixture Spatial-Spectral Transformer (MTB):
    ◦ Multi-head Feature Map Attention (MFMA): Mines spatial information.
    ◦ Multi-head Feature Channel Attention (MFCA): Mines spectral information.
    ◦ Multi-scale Convolutional Gated Feedforward Network (MCGFN): Effectively recovers local image structures through convolutions at different scales.
• Wavelet-based High-Frequency (WHF) Loss Function: Integrated into the total loss to enhance the network's ability to express image edges and recover sharpened high-frequency details.

🎯 Main Results

Experiments on three simulated datasets (CAVE, ICVL, Chikusei) and one real-world dataset (Hyperion-Sentinel-2) demonstrate that MIMO-SST surpasses existing state-of-the-art methods across multiple metrics (PSNR, RMSE, ERGAS, SAM, UIQI, SSIM).

On the CAVE dataset (upsampling factor=8):
• PSNR: 47.30 dB (approx. 0.85 dB improvement over the previous best method)
• Faster inference speed, with fewer computations (FLOPs) and parameters.

For details, please refer to Tables II-IV and XII in the paper.

🏗 Model Architecture

The overall network architecture is shown in the figure below (see Figure 2 in the paper for details):

<p align="center">
  <img src="docs/network_architecture.png" alt="MIMO-SST Architecture" width="800"/>
</p>

It mainly includes:
1. Mixture SST Block (MTB): The core module that fuses spatial and spectral information from LR-HSI and HR-MSI.
2. Multi-input Encoder: Processes input images at different downsampled scales to extract multi-scale features.
3. Multi-output Decoder: Generates multi-scale HR-HSI outputs in a coarse-to-fine manner.
4. Channel Transformer Block (CTB): Further explores spectral information within the encoder and decoder.

🚀 Quick Start

1. Environment Configuration
• Python >= 3.8
• PyTorch >= 2.0.0 (recommended 2.1.0)
• CUDA >= 11.7 (for GPU training)
• Other dependencies: numpy, scipy, tqdm, einops, pyyaml, tensorboard

You can set up the environment with the following commands:

Clone this repository

git clone https://github.com/Freelancefangjian/MIMO-SST.git
cd MIMO-SST

Create and activate a conda environment (optional)

conda create -n mimo-sst python=3.8
conda activate mimo-sst

Install PyTorch (adjust according to your CUDA version)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Install other dependencies

pip install -r requirements.txt

2. Dataset Preparation
We use three public datasets for training and evaluation:
• CAVE Dataset: http://www.cs.columbia.edu/CAVE/databases/multispectral/
• ICVL Dataset: http://icvl.cs.bgu.ac.il/hyperspectral/
• Chikusei Dataset: http://naotoyokoya.com/Download.html

Data Preprocessing:
1. Place the downloaded datasets into the ./datasets/ directory.
2. Run the provided preprocessing scripts to generate the required .mat or .h5 files for training and testing. Please refer to the scripts in the prepare_data/ directory and modify the paths as instructed.

   cd prepare_data
   # Modify the dataset paths in the scripts, then execute
   python prepare_cave.py
   python prepare_icvl.py
   python prepare_chikusei.py
   
3. The processed data will contain image patches of LR-HSI, HR-MSI, and the corresponding HR-HSI (Ground Truth).

3. Training the Model
Configuration files are located in the configs/ directory. You can modify hyperparameters (such as dataset paths, learning rate, batch size, etc.) as needed.
• Training on the CAVE dataset (upsampling factor 8):

  python train.py --config configs/train_cave_s8.yaml
  
• The training process supports TensorBoard logging. You can monitor training progress with:

  tensorboard --logdir ./logs
  

4. Testing and Evaluation
After training, use the test script to evaluate the model performance.
• Test the model:

  python test.py --config configs/test_cave_s8.yaml --checkpoint ./checkpoints/best_model_cave_s8.pth
  
• The script will compute and output metrics such as PSNR, RMSE, ERGAS, SAM, UIQI, SSIM, and optionally save the fused result images.

5. Using Pre-trained Models
We provide a pre-trained model trained on the CAVE dataset (upsampling factor 8). You can download it from the https://github.com/Freelancefangjian/MIMO-SST/releases page, or use the following command (if uploaded):

Assuming the model is managed via git-lfs

git lfs pull

Then, follow the testing steps above to load the pre-trained model for evaluation.

📁 Project Structure

MIMO-SST/
├── configs/             # Configuration files for training and testing (YAML format)
├── data/                # Data loading and processing modules
├── models/              # MIMO-SST network model definitions
│   ├── mimo_sst.py     # Main network architecture
│   ├── attention.py    # MFMA, MFCA modules
│   ├── feedforward.py  # MCGFN module
│   └── blocks.py       # MTB, CTB, and other basic blocks
├── losses/             # Loss function definitions (content loss, WHF loss)
├── trainers/            # Training loop logic
├── utils/               # Utility functions (metric calculation, logging, visualization)
├── prepare_data/       # Dataset preprocessing scripts
├── train.py            # Main training script
├── test.py             # Main testing script
├── requirements.txt    # Python dependency list
└── README.md           # This file


📈 Reproducing Experimental Results
The main quantitative results in the paper can be reproduced by following these steps:
1. Fully process the CAVE, ICVL, and Chikusei datasets as described in the "Dataset Preparation" section.
2. Train models for different upsampling factors using the respective config files: configs/train_*_s8.yaml, *_s16.yaml, *_s32.yaml.
3. Run test.py with the corresponding test configuration and the trained checkpoint.
4. Run utils/evaluate_all.py (if provided) to batch compute and generate averaged results consistent with the paper's tables.

🤝 Citation
If this project is helpful for your research, please cite our paper:

```bibtex
@article{fang2024mimosst,
  title={{MIMO-SST}: Multi-Input Multi-Output Spatial-Spectral Transformer for Hyperspectral and Multispectral Image Fusion},
  author={Fang, Jian and Yang, Jingxiang and Khader, Abdolraheem and Xiao, Liang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={62},
  pages={1--16},
  year={2024},
  doi={10.1109/TGRS.2024.3361553}
}
```

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

🙏 Acknowledgements
• Thanks to the providers of all public datasets (CAVE, ICVL, Chikusei).
• This work was partially supported by the National Natural Science Foundation of China (Grants 61871226, 61571230, 62001226), among other projects.

📮 Contact
If you have any questions or suggestions, feel free to contact us via:
• Author: Jian Fang (mailto:fangjian@njust.edu.cn)
• Submit an issue at the GitHub repository: https://github.com/Freelancefangjian/MIMO-SST/issues.

Please give us a star if you like this project! ⭐
