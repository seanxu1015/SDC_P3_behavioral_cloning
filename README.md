Behavioral Cloning

   At first approach, I implemented the model described in the given reference, and it worked well and the model was quite small, whereas it is a bit slow during tuning procedure. 
   I learned that inputs with size of 32*16 would also work in this project from forums. Thus I designed the following archecture, which has smaller image size but deeper channels. The fullly connected layers are just the same as the model in the given reference. This model runs two times faster than the reference model, however, the size is also merely two times bigger (1.8 mb vs 1.0 mb).
   
Archecture Information:

The model has three convolution layers and four fully connected layers with elu as activation function and dropout, excluding the last scoring layer. The details are shown below:
____________________________________________________________________________________________________ 
Layer Output Shape         Param #

convolution  (None, 32, 16, 32)   896         kernel=(3, 3)  stride=(1, 1)  activation='elu' 

maxpooling   (None, 32, 8, 16)     0

dropout      (None, 32, 8, 16)     0            

convolution  (None, 64, 8, 16)     18496       kernel=(3, 3) stride=(1, 1) activation='elu'     

maxpooling   (None, 64, 4, 8)      0          

dropout      (None, 64, 4, 8)      0                   

convolution  (None, 128, 4, 8)     24704       kernel=(3, 1) stride=(1, 1) activation='elu'  

dropout      (None, 128, 4, 8)     0                

flatten      (None, 4096)          0                          

dense        (None, 100)           409700      hidden_size=100 activation='elu'

dropout      (None, 100)           0                               

dense        (None, 50)            5050        hidden_size=50 activation='elu'             

dropout      (None, 50)            0                               

dense        (None, 10)            510         hidden_size=10 activation='elu'             

dropout      (None, 10)            0                               

dense        (None, 1)             11          hidden_size=1              
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

