# 🫀 Heart Disease Prediction Using Machine Learning and Deep Learning

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

## Introduction 

Predicting the condition of a patient in the case of __heart disease__ is really important. It would be good if a patient could get to know the condition before itself rather than visiting the doctor. Patients spend a __significant amount__ of time trying to get appointment from doctors. After they get the appointment, they would have to give a lot of tests. It is also important that a doctor be present so that they could treat them. To make things worse, the tests usually take a long time before __diagnosing__ whether a person would suffer from a heart disease. However, it would be handy and better if we automate this process which ensures that we save a lot of time and effort on the part of the doctor and patient. 

![Alt text](Image1.jpg) ![Alt text](images.jpg)

## Machine Learning and Data Science 

With the aid of machine learning, it is possible to get to know the condition of a patient whether he/she would have a heart disease or not based on a few features such as glucose levels and so on. Using machine learning algorithms, we are going to predict the chances of a person suffering from a heart disease. In other words, we would be using various machine learning models for the prediction of the chances of a heart disease in a patient. 

## Data 

Some of the input features that we would be considering in this example are __blood pressure__, __chest pain type__ and __cholestrol levels__. This is just a sample dataset with 303 rows. This is created just to understand different classification machine learning algorithms and sklearn implementation of them. Below is the link where you can find the source of the data. Thanks to __'kaggle'__ for providing this data. 

https://www.kaggle.com/johnsmith88/heart-disease-dataset

## Data Visualization 

There are many classification metrics that should be taken into consideration when performing the machine learning analysis. Considering the metrics, it is possible to evaluate the performance of different machine learning and deep learning models respectively. We would be tabulating the results for this particular dataset. We would be looking at various machine learning models and understanding their hyperparameters in the process so as to build the best version of the models.

In addition to this, we would also be plotting various graphs and understanding the relationships between various features in the dataset. We would first check if there are any null values in the data. After doing that step, we would then understand how the data is spread and their overall percentile values. We would be plotting the graphs and trying to see if there is any feature that could be reduced if there is a correlation. 

We would consider various important classification metrics for different machine learning models using the sklearn library from python. 
We would be using these tools to calculate the accuracy, precision and recall to name a few. For each machine learning model, we would then be drawing and tabulating the results at the end so that one would get a good idea about which algorithm did the best in each metric. One thing to keep in mind is that this is just a very small data set which is used just to understand the machine learning models and implement them easily. Therefore, one can go to use much more complex datasets by keeping these steps in mind while performing the execution. 

## Exploratory Data Analysis (EDA)

When we are performing EDA, interesting insights could be drawn from the data and understood. Below are the key insights that were found as a result of working with the data. 

* The features __'thalach'__ and __'slope'__ are positively correlated. However, they are not highly correlated.
* Features such as __'age'__ and __'thalach'__ which stands for __maximum heart rate achieved__ are negatively correlated as indicated by the correlation heatmap plot. 
* __'age'__ is negatively correlated with __'restecg'__ as shown in the heatmap plot.
* The features __'cp'__ and __'exang'__ are negatively correlated.
* The feature __'resting blood pressure'__ is somewhat positively correlated with __'age'__ as shown in the plots. 
* The feature __'trestbps'__ that stands for resting blood pressure is somewhat correlated with the feature __'fbs'__ that stands for fasting blood pressure.

## Machine Learning Models 

There were many machine learning models used in the process of predicting the __heart diseases__. Below are the models that were used in the process of predicting the heart diseases.

* __K Nearest Neighbors (KNN)__
* __Logistic Regression__
* __Naive Bayes__
* __Random Forest Classifier__

## Outcomes

* 

## 👉 Directions to download the repository and run the notebook 

This is for the Washington Bike Demand Prediction repository. But the same steps could be followed for this repository. 

1. You'll have to download and install Git that could be used for cloning the repositories that are present. The link to download Git is https://git-scm.com/downloads.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Heart-Disease-Prediction/blob/main/images/Screenshot%20(14).png" width = "600"/>
 
2. Once "Git" is downloaded and installed, you'll have to right-click on the location where you would like to download this repository. I would like to store it in "Git Folder" location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Heart-Disease-Prediction/blob/main/images/Screenshot%20(15).png" width = "600" />

3. If you have successfully installed Git, you'll get an option called "Gitbash Here" when you right-click on a particular location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Heart-Disease-Prediction/blob/main/images/Screenshot%20(16).png" width = "600" />


4. Once the Gitbash terminal opens, you'll need to write "Git clone" and then paste the link of the repository.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Heart-Disease-Prediction/blob/main/images/Screenshot%20(18).png" width = "600" />

5. The link of the repository can be found when you click on "Code" (Green button) and then, there would be a html link just below. Therefore, the command to download a particular repository should be "Git clone html" where the html is replaced by the link to this repository. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Heart-Disease-Prediction/blob/main/images/Screenshot%20(17).png" width = "600" />

6. After successfully downloading the repository, there should be a folder with the name of the repository as can be seen below.

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Heart-Disease-Prediction/blob/main/images/Screenshot%20(19).png" width = "600" />

7. Once the repository is downloaded, go to the start button and search for "Anaconda Prompt" if you have anaconda installed. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Heart-Disease-Prediction/blob/main/images/Screenshot%20(20).png" width = "600" />

8. Later, open the jupyter notebook by writing "jupyter notebook" in the Anaconda prompt. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Heart-Disease-Prediction/blob/main/images/Screenshot%20(21).png" width = "600" />

9. Now the following would open with a list of directories. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Heart-Disease-Prediction/blob/main/images/Screenshot%20(22).png" width = "600" />

10. Search for the location where you have downloaded the repository. Be sure to open that folder. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Heart-Disease-Prediction/blob/main/images/Screenshot%20(12).png" width = "600" />

11. You might now run the .ipynb files present in the repository to open the notebook and the python code present in it. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Heart-Disease-Prediction/blob/main/images/Screenshot%20(13).png" width = "600" />

That's it, you should be able to read the code now. Thanks. 
