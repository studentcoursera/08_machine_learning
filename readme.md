# Machine learning: Predict each classe using acceletometer data  
  
  
## Data for this project  
  
The training data  : https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv  
The test data      : https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv  
Source of the data : http://groupware.les.inf.puc-rio.br/har  
  
  
## Background  
  
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  
  
In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.   
  
More information accessed from http://groupware.les.inf.puc-rio.br/har (the section on the Weight Lifting Exercise Dataset).  
  
  
## References
  
Qualitative Activity Recognition of Weight Lifting Exercises
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th Augmented Human (AH) International Conference in cooperation with ACM SIGCHI (Augmented Human'13) . Stuttgart, Germany: ACM SIGCHI, 2013.
Read more: http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201#ixzz3oNTQE11n

Link: http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf
  
  
## Goal of this project  
  
0. We use data on the "Weight Lifting Exercise" Dataset and use accelerometers on the belt, forearm, arm, and dumbell of 6 participants.
1. To predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with.  
2. The report describes  
    a. how we built the model,  
	b. how we used cross validation,  
	c. what we think the expected out of sample error is, and  
	d. why we made the choices we did.  
3. We also use our prediction model to predict 20 different test cases.  
  
## Brief about the Weight lifting Exercises  
  
Participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: 
1. exactly according to the specification (Class A), 
2. throwing the elbows to the front (Class B), 
3. lifting the dumbbell only halfway (Class C), 
4. lowering the dumbbell only halfway (Class D) and 
5. throwing the hips to the front (Class E). 

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. 

The exercises were performed by 6 male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).

## Notes  
  
3 types of motion (Euler angels): roll, pitch and yaw.  
--------------------------------  
Rotation around the front-to-back axis is called roll.  
Rotation around the side-to-side axis is called pitch.  
Rotation around the vertical axis is called yaw.  
  
  
Variables  
---------  
Training and testing data:  
+ Totally, there are 160 fields.  
+ 100 fields are the aggregated values.  
+ 60 fields are independent values recorded.  
  
1. For every user, every num_window and new_window == "yes", 8 values are calculated. The combination of the same is as follows (96+4=100 fields):  
   a. belt, arm, forearm, dumbell (4 tools)  
      i. roll, pitch and yaw (3 motions/Euler angels)  
          a. kurtosis, skewness, max, min, amplitude, avg, stddev, var (8 calculated features)  
      ii. accel  
          a. var  
2. The values in every column would be (60 fields):  
   a. user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window [ 6]  
   b. belt, arm, forearm, dumbell (4 tools)															[12]  
      i. roll_<tools>, pitch_<tools> and yaw_<tools> (3 motions/Euler angels)  
   c. belt, arm, forearm, dumbell (4 tools)															[ 4]  
      i. total_accel_<tools>  
   d. belt, arm, forearm, dumbell (4 tools)															[36]  
      i. gyros, accel, magnet (3 sensors)  
        a. <sensors> _<tools>_x/y/z  
   e. X and classe/problem_id_																		[ 2]  
3. There are no NAs in main data (site http://groupware.les.inf.puc-rio.br/har). The NAs found in the training or test data, looks basically like, for the columns where the num_windows are same (new_window == "no"). So, we can remove those NAs.   
4. In training data there is a variable, "classe".  
   In testing, there is no "classe" but another field called "problem_id".  
5. In testing data,   
   a. there is not a single row with new_window="yes"; thus, there are no values for these fields (96+4=100) combinations mentioned in point 1. So, out of 160 fields, these 100 fields have no values; only "NAs".  
   b. So, the rest of 60 fields composes of - as detailed in point 2 above.  
6. For the purpose of this project, we only need data related to classe and accelerometers on the belt, forearm, arm, and dumbell of 6 participants.  
  
  
## Experiments/observations  
  
1. During EDA, noticed that particular set of variables had interesting patterns. One of them was for dumbbell, there was interesting pattern of a diamond shape. If you need to refer, refer to the Appendix.  
  
2. When we run random forests, without mentioning number of trees, it took almost 3+ hours to train (Where prox=TRUE). But, at the same time, if prox is not stated, notice that for the dataset with only accelerometer, it took about 10 minutes.  
  
3. Also, tried "lvq" (Learning Vector Quantization). This was conducted on 2 different datasets of eliminating 8 calculated features and the other x/y/z set of data sets. And it took between 2 to 3 hours to complete.  
  
4. After conducting "lvq", tried to get summarized importance of variables and noticed that gyros where the least important variables; for each classe.  
  
5. Could not use "allowParallel=TRUE" with "lvq", it throws error. Did not probe more as time was limited.
  
6. When used "Parallel Random Tree", it displayed a warning, stating parallelism was not used, it performed sequentially. In caret package, train function, method="parRF".  Warning message: executing %dopar% sequentially: no parallel backend registered
  
7. As part of high correlation test, noticed that couple of fields can be eliminated. This was experimented on the x/y/z/total dataset (43 variables); not on the final data.
```
> names(tt3[,highCorr])
[1] "accel_belt_x"    "gyros_belt_z"    "gyros_forearm_x" "magnet_arm_z"   
[5] "total_accel_arm"
```  
8. 
  
  
### Decision on final variables list  
  
After experimenting with the following group of variables and considering the current project requirement, finally decided to consider only variables related to accelerometer.  

The various forms of group of variables considered in this experimentaion (as part of EDA) were:
1. All the variables. This needed eliminating a few columns, like NA, etc.  
  
2. Eliminating all the 8 calculated variables. With this encountered a few issues.  
```
tt2 <- subset(pmltraining,select=c(-grep("^kurtosis_|^skewness_|^max_|^min_|^amplitude_|^avg_|^stddev_|^var_",names(pmltraining))))
```  
  
3. Considering roll/pitch/yaw/total and x/y/z variables for gyros/accel/magnet with classe. This was a fair consideration, it did train and predict well, as well.  
```
newTrain <- subset(pmltraining,select=c(grep("^total_|_x$|_y$|_z$|^roll_|^pitch_|^yaw",names(pmltraining)),grep("classe",names(pmltraining))))
```  
  
4. Considering total and x/y/z variables for gyros/accel/magnet with classe. This was a fair consideration, it did train and predict well, as well. This also performed well, could train and predict.  
```
tt <- subset(pmltraining,select=c(2,7,grep("^total_|_x$|_y$|_z$",names(pmltraining)),grep("classe",names(pmltraining))))
```  
  
5. First 7 and last 1 fields (8 fields), then, 16 fields of roll/pitch/yaw/total_accel for arm/belt/dumbbell/forearm, then, 36 fields of gyros/accel/magnet for arm/belt/dumbbell/forearm for x/y/z. [8 + 16 + 36 = 60 fields]  
```
pmltrain <- subset(pmltraining,c(1:7,grep("^roll_|^pitch_|^yaw_",names(pmltraining)),grep("^gyros_|^accel_|^magnet_|^total_accel",names(pmltraining)),grep("classe",names(pmltraining)))  
```  
  
6. Only username, num_window, accelerator variables and classe, for my own interpretation of data.  
```
pmltrain <- subset(pmltraining,select=c(2,7,grep("^accel_|^total_accel",names(pmltraining)),grep("classe",names(pmltraining))))
```  
  
7. Finally, filtered fields for only accelerometer and classe variable, as the project specifically calls this out.  
  

### Others

3 checks:
It is suggested that if:
1. the percentage of unique values is less than 20% and  
2. the ratio of the most frequent to the second most frequent value is greater than 20, the predictor may cause problem for some models. The function nearZeroVar can be used to identify near zero-variance predictors in a dataset. It returns an index of the column numbers that violate the two conditions above.  
3. some models are susceptible to multicollinearity (i.e., high correlations between predictors).  

Linear models, neural networks and other models can have poor performance in these situations or may generate unstable solutions. Other models, such as classification or regression trees, might be resistant to highly correlated predictors, but multicollinearity may negatively impact interpretability of the model.

So, if only there is need for 3rd check, do it. Else, do not perform.
[The above paragraph on checks, is with reference to "Journal of Statistical Software"; Building Predictive Models in R Using the caret Package by Max Kuhn]


### Experiment conducted on actual data -  8 calculated features: 
   
The 8 calculated features; some features have discrepancy:  
1. stddev, var, avg    : perfect, matches exactly.  
2. skewness, kurtosis  : some match very close to values, but not an exact match; so did not probe further.  
3. min, max, amplitude : are differently matched (swapped), for the ones stated below; the ones not listed below, do not correspond/match to any.  
```  
		original			swapped_to
		================================
		pitch_arm			yaw_arm
		pitch_dumbbell		yaw_dumbbell
		pitch_forearm		yaw_forearm
		roll_belt			yaw_belt
		roll_arm			pitch_arm
		roll_dumbbell		pitch_dumbbell
		roll_forearm		pitch_forearm
```  
  

==================  

