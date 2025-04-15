# Query-Driven Feature Learning for Cross-View Geo-Localization

[//]: # ([![Paper]&#40;https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg&#41;]&#40;https://arxiv.org/abs/XXXX.XXXXX&#41;)
[//]: # ([![License]&#40;https://img.shields.io/badge/License-Apache%202.0-blue.svg&#41;]&#40;https://opensource.org/licenses/Apache-2.0&#41;)

Official PyTorch implementation of the paper **"Query-Driven Feature Learning for Cross-View Geo-Localization"**. This repository contains all experiments code and pre-trained model on University-1652.
___
## 📦 Environment Setup

```bash
# Create conda environment (Python 3.10+ required)
conda create -n qdfl python=3.10 -y
conda activate qdfl

# Install repository
git clone https://github.com/Shuyu-Hu/QDFL.git
cd QDFL

# Install core dependencies
pip install -r requirements.txt
```
___
## 📥 Model Weight
Download pre-trained model on University-1652:
```bash
https://drive.google.com/file/d/1kbLEOsRoF8Q0e0RYYtbcH6za5R65KBE1/view?usp=drive_link
```

## 🏋️ Training
Data Preparation
Download University-1652 and SUES-200 datasets from their repository.

University-1652: https://github.com/layumi/University1652-Baseline

SUES-200: https://github.com/Reza-Zhu/SUES-200-Benchmark

Organize SUES-200 dataset using script in "your_path/QDFL/utils/sues_split_datasets.py":
```
├─ SUES-200
  ├── Training
    ├── 150/
    ├── 200/
    ├── 250/
    └── 300/
  ├── Testing
    ├── 150/
    ├── 200/ 
    ├── 250/	
    └── 300/
```
Then, make sure to update the dataset paths for the files in `datasets/train`.  

Start training using `main.py`.  

If you want to build your own model, add it to the `model` directory and register it in `get_backbone_components.py`.
## 📊 Evaluation
To evaluate on the test set, use `Supervised_evaluate.py`. 
Ensure that the config file and weight path are correct, then set `datasets_configs`.
```python
configs = load_config('./model_configs/dino_b_QDFL.yaml')["model_configs"]

pth_path = '/home/whu/Documents/codespace/Drone_Sat_Geo_Localization/Current_SOTA/DINO_QDFL_U1652.pth'

datasets_configs = {
                    'U1652': ['sat->drone', 'drone->sat'],
                    # 'DenseUAV':['drone->sat'],
                    'SUES200':([150, 200, 250, 300],['sat->drone','drone->sat'])
                    }
```
If you run a single test case, the result will be displayed in the terminal. If you use the "test_all" case, 
the results will be saved in the directory where your weights are stored.
## 🤝 Contact
For questions or collaborations:
```
Email: hushuyu@sia.cn

GitHub Issues: Open Issue
```

## 🙏 Acknowledgement
We gratefully acknowledge the following contributions that made this research possible:
* **Dataset Providers** of the University-1652 and SUES-200 dataset for their pioneering work in cross-view geo-localization benchmarking.
* **Our colleagues in the cross-view geo-localization research community** for their valuable insights and constructive discussions.
* **Reviewers and editors** whose feedback helped improve this work.

This repository is based on the excellent framework [Pytorch-Lightning](https://github.com/Lightning-AI/pytorch-lightning), and we also want to thank the creators of the excellent works:
* [DINOv2](https://github.com/facebookresearch/dinov2)
* [MixVPR](https://github.com/amaralibey/MixVPR)
* [SelaVPR](https://github.com/Lu-Feng/SelaVPR)