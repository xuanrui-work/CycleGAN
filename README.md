# CycleGAN

This repository contains the pytorch implementation of the following papers:
* [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

## Getting started

### Prerequisites
Run the below command to install all the required python dependencies:
```
pip install -r requirements.txt
```

### Usage
Simply open the corresponding jupyter notebook with your favorite editor and run the cells. Training logs and visualizations will be displayed through tensorboard, which can be started by running the below command:
```
tensorboard --logdir=<dir-to-logs>
```

### Results
![alt](./docs/images/results-1.png)
![alt](./docs/images/results-2.png)
