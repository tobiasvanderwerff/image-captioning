# image-captioning
This repository contains a PyTorch implementation of a deep learning based image-captioning system layed out in a 2015 paper from Xu et al. [1]. The system uses visual features from a CNN in combination with a LSTM language decoder and an attention mechanism to predict image captions. We include several different alternative CNN encoders: VGGnet, ResNet, MobileNet, and DenseNet.

## Instructions
First download the necessary packages, e.g. using pip:  

```
pip install -r requirements.txt
```

In order to run the experiments given in `hyperparams.yaml`, run

```
python train_model.py --data_path DIR
```

which will train a model on the Flickr8k dataset and show performance on the test and evaluation set. The --data_path argument specifies the directory where the dataset, log files and experiments will be stored. 

## References
[1] Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Rus-lan Salakhudinov, Rich Zemel, and Yoshua Bengio. Show, attend and tell: Neural  image  caption  generation  with  visual  attention.   In *International conference on machine learning*, pages 2048–2057. PMLR, 2015.
