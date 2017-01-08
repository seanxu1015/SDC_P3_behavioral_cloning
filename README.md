Behavioral Cloning

   At first approach, I implemented the model described in the given reference, and it worked well and the model was quite small, whereas it is a bit slow during tuning procedure. 
   I learned that inputs with size of 32*16 would also work in this project from forums. Thus I designed the following archecture, which has smaller image size but deeper channels. The fullly connected layers are just the same as the model in the given reference. This model runs two times faster than the reference model, however, the size is also merely two times bigger (1.8 mb vs 1.0 mb).
   
Archecture Information:

The model has three convolution layers and four fully connected layers with elu as activation function and dropout, excluding the last scoring layer. The details are shown below:
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 32, 16, 32)    896         convolution2d_input_1[0][0]      
kernel=(3, 3) stride=(1, 1) activation='elu'
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 32, 8, 16)     0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 32, 8, 16)     0           maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 64, 8, 16)     18496       dropout_1[0][0]                  
kernel=(3, 3) stride=(1, 1) activation='elu'
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 64, 4, 8)      0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 64, 4, 8)      0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 128, 4, 8)     24704       dropout_2[0][0]                  
kernel=(3, 1) stride=(1, 1) activation='elu'
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 128, 4, 8)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 4096)          0           dropout_3[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           409700      flatten_1[0][0]                  
hidden_size=100 activation='elu'
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_4[0][0]     
hidden_size=50 activation='elu'             
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_5[0][0]      
hidden_size=10 activation='elu'             
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_6[0][0]    
hidden_size=1              
____________________________________________________________________________________________________
Total params: 459,367
Trainable params: 459,367
Non-trainable params: 0
____________________________________________________________________________________________________

Training procedure:

1. Use the given dataset to tune parameters such as learning rate and weight decay so that the mse loss would converge.

2. Tune dropout to see whether the model could perform better on testing data.

3. Test the model on simulator, and find out where the model failed.

4. Collect new data by simulator to teach model how to recover from wrong actions where the model failed, and train the model again with these additional data.

5. Repeat procedure 3-4 until the vehicle could run well through the entire track. 

The dataset now is about two times bigger than the original one when the model succeeded. Most addition data is in the problem areas such as sharp turns, and some is to teach the vechile to recover from being too close to one side. 

Here are some data examples:

![image](https://github.com/seanxu1015/SDC_P3_behavioral_cloning/blob/master/images/center_2017_01_05_18_08_56_408.jpg)![image](https://github.com/seanxu1015/SDC_P3_behavioral_cloning/blob/master/images/center_2017_01_05_18_08_56_408.jpg)
![image](https://github.com/seanxu1015/SDC_P3_behavioral_cloning/blob/master/images/center_2017_01_05_18_08_56_408.jpg)![image](https://github.com/seanxu1015/SDC_P3_behavioral_cloning/blob/master/images/center_2017_01_05_18_08_56_408.jpg)

