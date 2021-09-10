## Available Datasets

Datasets can be downloaded in:
https://drive.google.com/drive/folders/1AThibyepa84PKW893XV0MwZyMoKy6B_e

To use the datasets with the WSEF framework, just extract them inside
the <em>"datasets/"</em> directory.

Currently available:
* Oxford17Flowers (https://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
* Corel5k (http://www.ci.gxnu.edu.cn/cbir/Dataset.aspx)
* Cub200-2011 (http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

Note: Features were extracted using:
* Resnet152: https://github.com/Cadene/pretrained-models.pytorch
* ACC: https://github.com/dermotte/LIRE

## Datasets Directory Structure

The <em>"datasets/"</em> directory contains a folder for each dataset, where the name must correspond to the dataset
name defined in the WSEF <em>config.py</em> file.
Inside each directory there is:
- A <em>groundtruth.txt</em> file that specifies the label of each class. This file uses the same format of the [UDLF classes file](https://github.com/UDLF/UDLF/wiki/File-Formats).
- The <em>rks</em> directory that contains a text file for each ranked list. The files use the same format of the [UDLF numeric ranked lists](https://github.com/UDLF/UDLF/wiki/File-Formats).
- The <em>features</em> directory that contains a .npz file for each feature. Each ranked list must have a corresponding feature.
Each file contains a list with a feature vector for each image.

```
datasets
│
└───flowers
│   │   groundtruth.txt
│   │
│   └───rks
│   │   │   acc.txt
│   |   │   resnet.txt
│   |   │   ...
│   │
│   └───features
│       │   acc.npz
│       │   resnet.npz
│       │   ...
│
└───corel5k
│   │   groundtruth.txt
│   │
│   └───rks
│   │   │   acc.txt
│   |   │   resnet.txt
│   |   │   ...
│   │
│   └───features
│       │   acc.npz
│       │   resnet.npz
│       │   ...
```
