# Biometric_Activity_Classifier 
Physical activity classification of biometric (IMU) data using machine learning   
Inertial measurement unit (IMU) data consists of accelerometer and gyroscope on x, y, z axises

## SVM Results  

### Confusion matrix: activity classifier from phone biometric data    
![confusion_matrix_phone](https://user-images.githubusercontent.com/33404359/156106066-25423b3b-d110-4d98-a83f-fab1a3936136.png)

### Confusion matrix: activity classifier from smartwatch biometric data    
![confusion_matrix_watch](https://user-images.githubusercontent.com/33404359/156106079-85e3cf51-7cce-4059-b498-658789886dd5.png)

## Analysis of phone biometric data   

### Accelerometer (m/s^2)  

#### Scatter plot matrix   
![PCA_walking_accel_phone_histogram](https://user-images.githubusercontent.com/33404359/155634931-c345e74d-b76a-4e16-b77d-9654a4b4462d.png)

#### Line graph first minute  
![walking_accel_phone](https://user-images.githubusercontent.com/33404359/155635541-917ad073-d19f-4bf1-8e90-8f5f28025cdf.png)


### Gyroscope (radians/s)  

#### Scatter plot matrix   
![PCA_walking_gyro_phone_histogram](https://user-images.githubusercontent.com/33404359/155634945-e2313805-9488-49ca-aa60-709adebc54a7.png)

#### Line graph first minute  
![walking_gyro_phone](https://user-images.githubusercontent.com/33404359/155635601-a5d4fda8-a0c5-4d05-9326-3ce75afb477f.png)


### Inertial measurement unit (IMU)  

#### Scatter plot matrix   
![PCA_walking_imu_watch_histogram](https://user-images.githubusercontent.com/33404359/155634958-5aea38eb-7c55-479a-822f-c6bb9b9636ba.png)

#### Line graph first minute  
![walking_imu_phone](https://user-images.githubusercontent.com/33404359/155635619-d558f847-ea82-4028-9c3c-fd08096e748d.png)


### Notes: 
Phone in pocket   
Watch on wrist of dominant hand   

20Hz sampling rate for phone and watch     
~64800 samples -> ~54 minutes of data for each subject      

Subject-id: unique to subject, Range: 1600-1650  
ActivityLabel: unique activity, Range: A-S (no “N” value)  
Timestamp: Integer, Linux time  
x: x axis of sensor   
y: y axis of sensor   
z: z axis of sensor   

# Data Set Used / link to study   
https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+
