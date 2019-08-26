# Skin-Cancer-Classification-Tflite-Model
skin cancer classification Tensorflow Lite model which can be integrated on Android and ios

Changes made in version 3

Added a Classification Report.
Added some text explaining the terms Precison, Recall and F1 score.
Introduction

I've always wanted to build an end to end ml solution - starting with model creation and ending with a live web app. Here I've managed to do it. Users are able to submit a picture of a skin lesion and get an instant prediction. This kernel details the process I followed to build the model and then convert it from Keras to Tensorflow.js. The javascript, html and css code for the app is available on github. 

Web App:
http://skin.test.woza.work/
Github: 
https://github.com/vbookshelf/Skin-Lesion-Analyzer

This model classifies skin lesions into seven classes. It is a fine tuned MobileNet CNN. All training was done in this kernel. The main challenges were the unbalanced dataset and the small amount of data. I used data augmentation to reduce the class imbalance and in so doing get categorical accuracy scores that were not heavily skewed by a single majority class.

MobileNet’s small size and speed makes it ideal for web deployment. It’s also a joy to train.

Tensorflow.js is a new library that allows machine learning models to run in the browser - without having to download or install any additional software. Because the model is running locally, any data that a user submits never leaves his or her pc or mobile phone. I imagine that privacy is especially important when it comes to medical data.

What is the objective?

I found it very helpful to define a clear objective right at the start. This helps guide the model selection process. For example, if a model has an accuracy of 60% it would usually be seen as a bad model. However, if it also has a top 3 accuracy of 90% and the objective requires that it output 3 predictions then it may actually be quite a good model.

This is the objective that I defined for this task:

Create an online tool that can tell doctors and lab technologists the three highest probability diagnoses for a given skin lesion. This will help them quickly identify high priority patients and speed up their workflow. The app should produce a result in less than 3 seconds. To ensure privacy the images must be pre-processed and analysed locally and never be uploaded to an external server.

from numpy.random import seed
seed(101)
from tensorflow import set_random_seed
set_random_seed(101)

import pandas as pd
import numpy as np
#import keras
#from keras import backend as K

import tensorflow
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import os

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
%matplotlib inline
LABELS

Excerpts from the paper:

The HAM10000 Dataset: A Large Collection of Multi-Source Dermatoscopic Images of Common Pigmented Skin Lesions
https://arxiv.org/abs/1803.10417

nv
Melanocytic nevi are benign neoplasms of melanocytes and appear in a myriad of variants, which all are included in our series. The variants may differ significantly from a dermatoscopic point of view.
[6705 images]

mel
Melanoma is a malignant neoplasm derived from melanocytes that may appear in different variants. If excised in an early stage it can be cured by simple surgical excision. Melanomas can be invasive or non-invasive (in situ). We included all variants of melanoma including melanoma in situ, but did exclude non-pigmented, subungual, ocular or mucosal melanoma.
[1113 images]

bkl
"Benign keratosis" is a generic class that includes seborrheic ker- atoses ("senile wart"), solar lentigo - which can be regarded a flat variant of seborrheic keratosis - and lichen-planus like keratoses (LPLK), which corresponds to a seborrheic keratosis or a solar lentigo with inflammation and regression [22]. The three subgroups may look different dermatoscop- ically, but we grouped them together because they are similar biologically and often reported under the same generic term histopathologically. From a dermatoscopic view, lichen planus-like keratoses are especially challeng- ing because they can show morphologic features mimicking melanoma [23] and are often biopsied or excised for diagnostic reasons.
[1099 images]

bcc
Basal cell carcinoma is a common variant of epithelial skin cancer that rarely metastasizes but grows destructively if untreated. It appears in different morphologic variants (flat, nodular, pigmented, cystic, etc) [21], which are all included in this set.
[514 images]

akiec
Actinic Keratoses (Solar Keratoses) and intraepithelial Carcinoma (Bowen’s disease) are common non-invasive, variants of squamous cell car- cinoma that can be treated locally without surgery. Some authors regard them as precursors of squamous cell carcinomas and not as actual carci- nomas. There is, however, agreement that these lesions may progress to invasive squamous cell carcinoma - which is usually not pigmented. Both neoplasms commonly show surface scaling and commonly are devoid of pigment. Actinic keratoses are more common on the face and Bowen’s disease is more common on other body sites. Because both types are in- duced by UV-light the surrounding skin is usually typified by severe sun damaged except in cases of Bowen’s disease that are caused by human papilloma virus infection and not by UV. Pigmented variants exists for Bowen’s disease [19] and for actinic keratoses [20]. Both are included in this set.
[327 images]

vasc
Vascular skin lesions in the dataset range from cherry angiomas to angiokeratomas [25] and pyogenic granulomas [26]. Hemorrhage is also included in this category.
[142 images]

df
Dermatofibroma is a benign skin lesion regarded as either a benign proliferation or an inflammatory reaction to minimal trauma. It is brown often showing a central zone of fibrosis dermatoscopically [24].
[115 images]


[Total images = 10015]


Tensorflow model is created then it is converted Tflite model through TOCO object to be integrated On Android.
