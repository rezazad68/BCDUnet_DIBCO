# [DIBCO: Document Image Binarization Competition using BCDU-Net to achieve best performance](https://vc.ee.duth.gr/dibco2019/)

This github page contains my implementation for DIBCO challenges in Document Image Binarization. The implemented code uses BCDU-Net model for learning binarization process on DIBCO series. The evaluation results shows the BCDU-net chan achieve the best performance on DIBCO challenges. If this code helps with your research please consider citing the following paper:
</br>
> [R. Azad](https://scholar.google.com/citations?hl=en&user=Qb5ildMAAAAJ&view_op=list_works&sortby=pubdate), et. all, "Bi-Directional ConvLSTM U-Net with Densely Connected Convolutions ", ICCV, 2019, download [link](https://arxiv.org/pdf/1909.00166.pdf).

## Updates
- December 3, 2019: First release (Complete implemenation for [DIBCO Series](https://vc.ee.duth.gr/dibco2019/), years 2009 to 2017 datasets added.). Other dataset can be add easily.

## Prerequisties and Run
This code has been implemented in python language using Keras libarary with tensorflow backend and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:

- Python 3
- Keras - tensorflow backend


## Run Demo
For training deep model for each DIBCO year, follow the bellow steps:

#### DIBCO Series
1- Download the DIBCO datasets from [this](https://challenge.kitware.com/#phase/5abcb19a56357d0139260e53) link and extract it. We included DIBCO datasets from 2009 to 2017. It is easy to add DIBCO 2018, 2019 or other dataset, just need to revise the utils code. </br>
2- Run `Prepare_DIBCO.py` for data preperation and dividing data to train and test sets. Please note that this code will consider whole the samples of one particular year as a test set and rest of the years for the training set. It is the common data division which uses in DIBCO challenge. </br>
3- Run `Train_DIBCO.py` for training BCDU-Net model using trainng and validation (20% of the training samples) sets. The model will be train for 100 epochs and it will save the best weights for the valiation set. </br>
4- For performance calculation and producing binarization result, run `Evaluate.py`. It will represent performance measures and will saves related figures and results in `output` folder.</br>
Notice: We train the model using patches that we extract from the training set. Also for test image binarization we apply patch-based overlaping binarization. Similiart to the approach we used in medical Retina image.


## Quick Overview

### Structure of the Bidirection Convolutional LSTM that used in BCDU-Net network
![Diagram of the proposed method](https://github.com/rezazad68/LSTM-U-net/blob/master/output_images/convlstm.png)

## Results
For evaluating the performance of the BCDU-Net model, we followed the resutl of DIBCO 2014 and DIBCO 2016 have been considered for evaluation. In bellow, results of the proposed approach illustrated.
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
