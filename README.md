# [DIBCO: Document Image Binarization Competition using BCDU-Net to achieve best performance](https://vc.ee.duth.gr/dibco2019/)

This github page contains my implementation for DIBCO challenges in Document Image Binarization. The implemented code uses BCDU-Net model for learning binarization process on DIBCO series. The evaluation results shows the BCDU-net chan achieve the best performance on DIBCO challenges.If this code helps with your research please consider citing the following paper:
</br>
> [R. Azad](https://scholar.google.com/citations?hl=en&user=Qb5ildMAAAAJ&view_op=list_works&sortby=pubdate), et. all, "Bi-Directional ConvLSTM U-Net with Densely Connected Convolutions ", ICCV, 2019, download [link](https://arxiv.org/pdf/1909.00166.pdf).

## Updates
- December 3, 2019: First release (Complete implemenation for [DIBCO Series](https://vc.ee.duth.gr/dibco2019/), years 2009 to 2017 dataset added.). Code simply can be adapted for other datasets too. Page will be updated in next few days

## Prerequisties and Run
This code has been implemented in python language using Keras libarary with tensorflow backend and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:

- Python 3
- Keras - tensorflow backend


## Run Demo
For training deep model for each DIBCO year, follow the bellow steps:

#### DIBCO Series
1- Download the ISIC 2018 train dataset from [this](https://challenge.kitware.com/#phase/5abcb19a56357d0139260e53) link and extract both training dataset and ground truth folders inside the `dataset_isic18`. </br>
2- Run `Prepare_ISIC2018.py` for data preperation and dividing data to train,validation and test sets. </br>
3- Run `train_isic18.py` for training BCDU-Net model using trainng and validation sets. The model will be train for 100 epochs and it will save the best weights for the valiation set. You can also train U-net model for this dataset by changing model to unet, however, the performance will be low comparing to BCDU-Net. </br>
4- For performance calculation and producing segmentation result, run `evaluate.py`. It will represent performance measures and will saves related figures and results in `output` folder.</br>


## Quick Overview
![Diagram of the proposed method](https://github.com/rezazad68/LSTM-U-net/blob/master/output_images/bcdunet.png)

### Structure of the Bidirection Convolutional LSTM that used in our network
![Diagram of the proposed method](https://github.com/rezazad68/LSTM-U-net/blob/master/output_images/convlstm.png)

## Results
For evaluating the performance of the BCDU-Net model, DIBCO 2014 and DIBCO 2016 have been considered for evaluation. In bellow, results of the proposed approach illustrated.
</br>
 

Methods | DIBCO 2014 |DIBCO 2016
------------ | -------------|----



Azad et. all [BCDU-Net](https://github.com/rezazad68/LSTM-U-net/edit/master/README.md)	 | **0.8222**	|**0.8012**


#### Document Image Binarization Results on DIBCO Series

![Documnet Image Binarization result 1](https://github.com/rezazad68/BCDUnet_DIBCO/blob/master/images/1.png)
![Documnet Image Binarization result 2](https://github.com/rezazad68/BCDUnet_DIBCO/blob/master/images/2.png)
![Documnet Image Binarization result 3](https://github.com/rezazad68/BCDUnet_DIBCO/blob/master/images/3.png)
![Documnet Image Binarization result 4](https://github.com/rezazad68/BCDUnet_DIBCO/blob/master/images/4.png)



### Query
All implementation done by Reza Azad. For any query please contact us for more information.

```python
rezazad68@gmail.com

```
