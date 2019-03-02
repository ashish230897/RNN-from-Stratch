# Building Your Own Deep Recurrent Neural Network
This repository does the same job as [this](https://github.com/ashish230897/Deep-Neural-Networks-From-Scratch) i.e it will walk you through the process of creating a **Deep Recurrent Neural Network** without using any of the deep learning *libraries* like *Tensorflow*, *Keras*, *Caffe* etc.  


The above repository shows how to implement the Recurrent Neural Network's forward propagation, back propagation from *stratch*.  
The **dinos.txt** file contains the names of dinosaurs. This is the *training* data that the model is trained upon.
The model learns *how to generate dinosaur's name given first few characters.*  
Backpropagation through time is implemented by adding the gradients across all the time steps.  
The best practices involved in training a good model is not followed as it is not the purpose of this repository.  
                                                                                                          
                                                                                                   
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
                                                             
                                                                       
### Prerequisites
The libraries required for the above project are : numpy(*Yes only Numpy :) *).  
                                                                          
                                                             
### Installing 
```
pip install numpy  
```  
                                                                   
                                                   
## Training the Model
The project above follows the following pattern while training the model:  
* Load the data
* Initialize parameters
* Forward Propagation
* Compute Cost
* Backward Propagation
* Update Parameters  
                           
The file *main.py* is to be run to start the training.  


## Acknowledgements
* Inspired by Andrew Ng!
* Got the code support from deeplearning.ai fifth course.

**In case of any *doubts/confusions* do shoot a mail at : ashish.agrawal2123@gmail.com**


