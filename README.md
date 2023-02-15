## Introduction

In semantic segmentation, background samples provide key contextual information for segmenting regions of interest (ROIs). Empirically, we find human-defined context labels (e.g., liver, kidney, brain tissue) provide additional anatomical information to benefit representation learning. For example, in the case of liver tumor segmentation, it is beneficial to also have labels for the liver available in addition to the tumor class. 

In this study, we further propose context label learning (CoLab), which automatically generates context labels to improve the learning of a context representation yielding better ROI segmentation accuracy based on a meta-learning scheme. CoLab can bring similar improvements when compared with training with human-defined context labels, without the need for expert knowledge.

<br/> <div align=center><img src="figs/MethodOverview.png" width="700px"/></div>

## Requirements

This code was developed with `python==3.7`.

```
ipykernel==6.9.1
simpleitk==2.2.1
scipy==1.2.1
scikit-image==0.17.2
matplotlib==3.0.3
torch==1.2.0
torchvision==0.4.0
tensorboard_logger==0.1.0
tensorboard
tensorflow==1.13.1
protobuf==3.15.8
nibabel==2.4.1
future==0.18.2
threadpoolctl==2.1.0
```

## Data and preprocessing

We conduct experiments with several medical image segmentation datasets. The datasets of liver tumor, colon tumor and pancreas tumor from [Medical Segmentation Decathlon](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2). Brain lesion dataset can be downloaded from [ATLAS](http://fcon_1000.projects.nitrc.org/indi/retro/atlas.html). Kidney tumor dataset can be downloaded from [KiTS19](https://github.com/neheller/kits19). We keep the downloaded data in `./datadownloaded`.

We provide the pre-processing code for liver tumor segmentation, which is based on CT. Please refer to `datapreprocessing.ipynb` about intensity normalization, resampling etc. After obtaining the preprocessed data saved in `./datapreprocessed`, we generate the datafiles such as those in `./datafiles`.

## Training

Train the network without context labels.

```
python train.py --name CoLab_LiverTumor_Vanilla --tensorboard --split 0 --deepsupervision --taskcls 1 --liver0 0 --taskupdate 5 --vanilla --det --gpu 0
```

Training with human-defined labels. Note that it utilized additional liver masks.

```
python train.py --name CoLab_LiverTumor_ManualLabel --tensorboard --split 1 --deepsupervision --taskcls 2 --liver0 0  --manuallabel --det --gpu 0
```

Traing with CoLab. Please refer to the paper for more hyperparamter details.

```
python train.py --name CoLab_LiverTumor --tensorboard --split 0 --deepsupervision --taskcls 2 --liver0 0 --taskupdate 5 --distdetach --threshold_sub 30 --threshold_dev 20 --det --gpu 0
```

## Test

Test the trained model. Please remember to replace `SAVE_PATH` with the ones you save the trained model. You might also use this to visualize the generated context labels by CoLab.

Note that if you want to test you models trained without context labels, you should set `taskcls = 1`.

```
python test.py --resume SAVE_PATH/checkpoint.pth.tar --name CoLab_LiverTumor_test --liver0 0 --saveresults --taskcls 2 --deepsupervision --gpu 0
```


## Citation

```
@article{li2023context,
  title={Context Label Learning: Improving Background Class Representations in Semantic Segmentation},
  author={Li, Zeju and Kamnitsas, Konstantinos and Ouyang, Cheng and Chen, Chen and Glocker, Ben},
  journal={IEEE Transactions on Medical Imaging},
  year={2023},
  publisher={IEEE}
}
```