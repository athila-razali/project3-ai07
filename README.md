# Identify Concrete Cracks with Image Classification
## Summary
**Aim** : To perform image classification to classify concretes with or without cracks.        
The problem is modelled as a binary classification problem : 1.Negative (no cracks); 2.Positive (has cracks)        
**Data Source** : [Mendeley Data](https://data.mendeley.com/datasets/5y9wdsg2zt/2)           
The dataset contains concrete images having cracks. 
Each class (Negative/Positive) has 20000 images with a total of 40000 images with 227 x 227 pixels with RGB channels.         
**Citation** :        
2018 – Özgenel, Ç.F., Gönenç Sorguç, A. “Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings”, ISARC 2018, Berlin.             

Lei Zhang , Fan Yang , Yimin Daniel Zhang, and Y. J. Z., Zhang, L., Yang, F., Zhang, Y. D., & Zhu, Y. J. (2016). Road Crack Detection Using Deep Convolutional Neural Network. In 2016 IEEE International Conference on Image Processing (ICIP). http://doi.org/10.1109/ICIP.2016.7533052

## Methodology
### 1. Data Pipeline
The image data are loaded and preprocessed. The data is first split into train-validation set, with a ratio of 70:30. 
The validation data is then further split into two portion to obtain some test data, with a ratio of 80:20.
The overall train-validation-test split ratio is 70:24:6. Data is ready to be trained.

### 2. Model pipeline
In TensorFlow, Data Augmentation is include as a part of model. Then to proceed with the model, the Feature Extraction and Classification layers is included.       
     
The deep learning model for this project is built using transfer learning. 
First, a preprocessing layer is made to convert the input images' pixel values to a range between -1 and 1.
In addition to acting as the feature scaler, this layer is necessary for the transfer learning model to generate the correct signals.
       
A MobileNet v2 pretrained model has been utilized for the feature extractor. 
The model comes pre-trained with ImageNet parameters and is easily accessible within the TensorFlow Keras package. 
Additionally, it is frozen, so updates won't be made during model training.
        
Softmax signals are produced using a classifier that uses a dense layer and global average pooling. The predicted class is recognised using the softmax signals.
        
The figure below shows a simplified illustration of the model.
        
![Untitled Diagram drawio (1)](https://user-images.githubusercontent.com/91872382/182074020-28388c1a-e742-4f18-a356-5ea4763f3e53.png)

The model evaluation before training is              
<img width="411" alt="image" src="https://user-images.githubusercontent.com/91872382/182075394-6219a7b9-c1eb-4daf-afe5-3443c23872d3.png">
          
The model is trained with a batch size of 16 and 10 epochs.
After training, the model reaches 99% training accuracy and validation accuracy. With help of TensorBoard, training results are shown in the figures below:       
<img width="262" alt="image" src="https://user-images.githubusercontent.com/91872382/182075662-dcab09ed-f806-437a-9c92-36eb806bbe2a.png"> <img width="257" alt="image" src="https://user-images.githubusercontent.com/91872382/182075736-9c1984bb-85ce-4db3-8b2d-bc6fa506182e.png">
          
## Results
The model evaluation after training is           
<img width="385" alt="image" src="https://user-images.githubusercontent.com/91872382/182075995-299e7e0e-7aeb-4a09-ab31-b7101c3222a8.png">
          
The model accuracy leads to a good prediction as shown in the figure below:            
*first column: label/actual values* ; *second column: prediction values*        
*0 = Negative; 1 = Positive*                
<img width="252" alt="image" src="https://user-images.githubusercontent.com/91872382/182076498-8940028f-5e28-45da-b23e-31427c6249cb.png">

### Credits
**Instructor : Kong Kah Chun                    
Selangor Human Resource Development Center**
