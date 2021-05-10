# Human_Action_recognition

Human activity recognition, or HAR for short, is a broad field of study concerned with identifying
the specific movement or action of a person based on sensor data.

Movements are often typical activities performed indoors, such as walking, talking, standing, and sitting. 
They may also be more focused activities such as those types of activities performed in a kitchen or on a factory floor.

The sensor data may be remotely recorded, such as video, radar, or other wireless methods.
Alternately, data may be recorded directly on the subject such as by carrying custom hardware or smart phones that have accelerometers and gyroscopes.

**The link bellow contains a dataset for our model:**
 - **This dataset contains 11 folders for different activities, each folder contains allmost 2500 images**

       https://drive.google.com/u/0/uc?export=download&confirm=-fln&id=1TPHNa6iwwJr0i9JmzN-c3YAf_tn7cwKz
       
       


**Here are the steps we will perform:**

- **Step 1: Download and Extract the Dataset**
- **Step 2: Visualize the Data with its Labels**
- **Step 3: Read and Preprocess the Dataset**
- **Step 4: Split the Data into Train and Test Set**
- **Step 5: Construct the Model**
- **Step 6: Compile & Train the Model**
- **Step 7: Plot Model’s Loss & Accuracy Curves**
- **Step 8: Make Predictions with the Model**


# The model structure

 ![model_structure_plot](https://user-images.githubusercontent.com/71146628/117590953-2d371d80-b132-11eb-92ac-013faa43e0b4.png)


# Model evaluation results 

- **Total Loss vs Total Validation Loss **

![Total Loss vs Total Validation Loss](https://user-images.githubusercontent.com/71146628/117591052-cbc37e80-b132-11eb-8a3c-57fe3f5fae84.png)

-**Total Accuracy vs Total Validation Accuracy **

![Total Accuracy vs Total Validation Accuracy](https://user-images.githubusercontent.com/71146628/117591034-a46cb180-b132-11eb-881c-b308a6de45d3.png)



## Model prediction output samples 

![image](https://user-images.githubusercontent.com/71146628/117591499-16de9100-b135-11eb-8ffb-c505d3b4313c.png)

![image](https://user-images.githubusercontent.com/71146628/117591573-5e651d00-b135-11eb-81d2-b94bd027b9be.png)




# Machine learning Model Deployment using Flask

## Introduction to Flask API
Flask is a lightweight Web Server Gateway Interface(WSGI) a micro-framework written in python. This means flask provides us with tools, libraries and technologies that allow us to build a web application. This web application can be some web pages, a blog, or our machine learning model prediction web application. Flask is an intermediate medium to connect our model with front end web page for prediction as shown in below image.

![image](https://user-images.githubusercontent.com/71146628/117591297-345f2b00-b134-11eb-94c9-dd5e3b13757f.png)


## Prerequisites

We assume that all of us have knowledge about model training in jupyter notebook. This post is aimed to only provide insights on deploying a machine learning model into production using Flask API.

**Libraries that require in model Deployment:**

    1.pip install pickle-mixin
    2.pip install Flask
    
- pickle: A native python library to save (serialize) and load (de-serialize) python objects as files on the disk.
- Flask: A python-based easy to use micro web framework.


# Our model web App

  - The main page 

![ML_App_mainPage](https://user-images.githubusercontent.com/71146628/117591620-99ffe700-b135-11eb-9032-4fa96def0142.jpeg)


  - The classification page

![image](https://user-images.githubusercontent.com/71146628/117591586-7341b080-b135-11eb-821a-66fca09fdaf0.png)

     
  
  



