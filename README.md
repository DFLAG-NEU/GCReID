# GCReID
Implementation of our [GCReID: Generalized continual person re-identification via meta learning and knowledge accumulation]([https://www.sciencedirect.com/science/article/pii/S089360802300045X](https://www.sciencedirect.com/science/article/pii/S0893608024004854)) in Neural Networks(2024) by Zhaoshuo Liu, Chaolu Feng, Kun Yu, Jun Hu and Jinzhu Yang.

# Install
## Enviornment
conda create -n creid python=3.7  
source activate creid  
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.0 -c pytorch  
conda install opencv  
pip install Cython sklearn numpy prettytable easydict tqdm matplotlib  

## Dataset prepration
Please follow [Torchreid_Dataset_Doc](https://kaiyangzhou.github.io/deep-person-reid/datasets.html) to download datasets and unzip them to your data path (we refer to 'machine_dataset_path' in train_test.py). Alternatively, you could download some of never-seen domain datasets in [DualNorm](https://github.com/BJTUJia/person_reID_DualNorm).

## Train & Test
python train_test.py

# Citation
@article{liu2024gcreid,
  title={GCReID: Generalized continual person re-identification via meta learning and knowledge accumulation},
  author={Liu, Zhaoshuo and Feng, Chaolu and Yu, Kun and Hu, Jun and Yang, Jinzhu},
  journal={Neural Networks},
  volume={179},
  pages={106561},
  year={2024},
  publisher={Elsevier}
}

