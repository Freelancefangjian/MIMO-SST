MIMO-SST (Multi-Input Multi-Output Spatial-Spectral Transformer)

https://img.shields.io/badge/PyTorch-2.0.0+-red](https://pytorch.org/)
https://img.shields.io/badge/Python-3.8+-blue]()
https://img.shields.io/badge/License-MIT-green]()
https://img.shields.io/badge/Paper-IEEE%20TGRS%202024-yellow](https://doi.org/10.1109/TGRS.2024.3361553)

官方 PyTorch 实现 论文 https://arxiv.org/abs/xxxx.xxxxx
 

本项目是高光谱与多光谱图像融合（HSI-MSI Fusion）任务的开源代码实现，对应论文《MIMO-SST: Multi-Input Multi-Output Spatial-Spectral Transformer for Hyperspectral and Multispectral Image Fusion》（IEEE TGRS 2024）。

📌 项目简介

本仓库提供了 MIMO-SST 网络的 PyTorch 实现。该网络是一种新颖的多输入多输出空间-光谱Transformer架构，旨在解决高光谱图像（HSI）超分辨率问题，即融合低空间分辨率的高光谱图像（LR-HSI） 和高空间分辨率的多光谱图像（HR-MSI），以生成高空间-光谱分辨率的高光谱图像（HR-HSI）。

核心创新点：
• 多输入多输出（MIMO）框架：通过从粗到细的架构，在不同尺度上监督HR-HSI的生成，从而更全面地捕捉图像细节和结构。

• 混合空间-光谱Transformer（Mixture Spatial-Spectral Transformer, MTB）：

    ◦ 多头特征图注意力（MFMA）：挖掘空间信息。

    ◦ 多头特征通道注意力（MFCA）：挖掘光谱信息。

    ◦ 多尺度卷积门控前馈网络（MCGFN）：通过不同尺度的卷积有效恢复局部图像结构。

• 小波高频（WHF）损失函数：集成到总损失中，以增强网络对图像边缘的表达能力，恢复锐化的高频细节。

🎯 主要结果

在三个模拟数据集（CAVE, ICVL, Chikusei）和一个真实数据集（Hyperion-Sentinel-2）上的实验表明，MIMO-SST 在多项指标上（PSNR, RMSE, ERGAS, SAM, UIQI, SSIM）超越了现有的先进方法。

在CAVE数据集上（上采样因子=8）：
• PSNR: 47.30 dB (比之前最佳方法提升约 0.85 dB)

• 更快的推理速度，更少的计算量（FLOPs）和参数量。

详情请参见论文中的表 II-IV 和 XII。

🏗 模型架构

网络整体架构如下图所示（详见论文图2）：
<p align="center">
  <img src="docs/network_architecture.png" alt="MIMO-SST Architecture" width="800"/>
</p>

主要包括：
1.  混合SST块（MTB）：核心模块，融合LR-HSI和HR-MSI的空间与光谱信息。
2.  多输入编码器：处理不同下采样尺度的输入图像，提取多尺度特征。
3.  多输出解码器：以从粗到细的方式生成多尺度的HR-HSI输出。
4.  通道Transformer块（CTB）：在编码器和解码器中进一步挖掘光谱信息。

🚀 快速开始

1. 环境配置

• Python >= 3.8

• PyTorch >= 2.0.0 (推荐 2.1.0)

• CUDA >= 11.7 (如需GPU训练)

• 其他依赖库：numpy, scipy, tqdm, einops, pyyaml, tensorboard

您可以通过以下命令安装环境：
# 克隆本仓库
git clone https://github.com/Freelancefangjian/MIMO-SST.git
cd MIMO-SST

# 创建并激活conda环境（可选）
conda create -n mimo-sst python=3.8
conda activate mimo-sst

# 安装PyTorch (请根据您的CUDA版本调整)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt


2. 数据集准备

我们使用三个公共数据集进行训练和评估：
• CAVE Dataset: http://www.cs.columbia.edu/CAVE/databases/multispectral/

• ICVL Dataset: http://icvl.cs.bgu.ac.il/hyperspectral/

• Chikusei Dataset: http://naotoyokoya.com/Download.html

数据预处理：
1.  将下载的数据集放入 ./datasets/ 目录下。
2.  运行提供的预处理脚本，生成训练和测试所需的 .mat 或 .h5 文件。请参考 prepare_data/ 目录下的脚本，并按照其中的说明修改路径。
    cd prepare_data
    # 修改脚本中的数据集路径，然后运行
    python prepare_cave.py
    python prepare_icvl.py
    python prepare_chikusei.py
    
3.  处理后的数据将包含LR-HSI、HR-MSI和对应的HR-HSI（Ground Truth）图像块。

3. 训练模型

配置文件位于 configs/ 目录。您可以根据需要修改超参数（如数据集路径、学习率、批大小等）。

• 在CAVE数据集上训练（上采样因子8）：
    python train.py --config configs/train_cave_s8.yaml
    
• 训练过程支持TensorBoard日志记录。您可以使用以下命令监控训练进度：
    tensorboard --logdir ./logs
    

4. 测试与评估

训练完成后，使用测试脚本评估模型性能。

• 测试模型：
    python test.py --config configs/test_cave_s8.yaml --checkpoint ./checkpoints/best_model_cave_s8.pth
    
• 脚本将计算并输出PSNR, RMSE, ERGAS, SAM, UIQI, SSIM等指标，并可选保存融合结果图像。

5. 使用预训练模型

我们提供了在CAVE数据集（上采样因子8）上训练的预训练模型。您可以从 https://github.com/Freelancefangjian/MIMO-SST/releases 页面下载，或使用以下命令（如果已上传）：
# 假设模型已通过git-lfs管理
git lfs pull

然后，按照上述测试步骤加载预训练模型进行评估。

📁 项目结构


MIMO-SST/
├── configs/               # 训练和测试的配置文件 (YAML格式)
├── data/                  # 数据加载和处理模块
├── models/                # MIMO-SST网络模型定义
│   ├── mimo_sst.py       # 主网络架构
│   ├── attention.py      # MFMA, MFCA 模块
│   ├── feedforward.py    # MCGFN 模块
│   └── blocks.py         # MTB, CTB 等基础块
├── losses/                # 损失函数定义 (内容损失, WHF损失)
├── trainers/              # 训练循环逻辑
├── utils/                 # 工具函数 (指标计算, 日志记录, 可视化)
├── prepare_data/          # 数据集预处理脚本
├── train.py              # 主训练脚本
├── test.py               # 主测试脚本
├── requirements.txt      # Python依赖包列表
└── README.md            # 本文件


📈 实验结果复现

论文中的主要定量结果可以通过以下步骤复现：
1.  按照“数据集准备”部分完整处理CAVE、ICVL、Chikusei数据集。
2.  分别使用 configs/train_*_s8.yaml, *_s16.yaml, *_s32.yaml 配置文件训练不同上采样因子的模型。
3.  使用对应的测试配置和训练好的检查点运行 test.py。
4.  运行 utils/evaluate_all.py（如果提供）可批量计算并生成与论文表格一致的均值结果。

🤝 引用

如果本项目对您的研究有所帮助，请引用我们的论文：
@article{fang2024mimosst,
  title={{MIMO-SST}: Multi-Input Multi-Output Spatial-Spectral Transformer for Hyperspectral and Multispectral Image Fusion},
  author={Fang, Jian and Yang, Jingxiang and Khader, Abdolraheem and Xiao, Liang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={62},
  pages={1--16},
  year={2024},
  doi={10.1109/TGRS.2024.3361553}
}


📄 许可证

本项目采用 MIT 许可证。详情请见 LICENSE 文件。

🙏 致谢

• 感谢所有公共数据集（CAVE, ICVL, Chikusei）的提供者。

• 本工作部分得到中国国家自然科学基金（Grants 61871226, 61571230, 62001226）等项目的资助。

📮 联系

如有任何问题或建议，欢迎通过以下方式联系：
• 作者：Jian Fang (mailto:fangjian@njust.edu.cn)

• 在GitHub仓库中提交 https://github.com/Freelancefangjian/MIMO-SST/issues。

如果喜欢这个项目，请给我们一个Star！⭐
