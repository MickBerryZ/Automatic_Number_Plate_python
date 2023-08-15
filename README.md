# Automatic_Number_Plate_python

This automatic Number Plate uses Python to implement that supports the English language.
It can detect and recognise number plates into CSV file.

Download VDO that uses this code : https://drive.google.com/file/d/1YmHTElM6rh5uBpvaoUYpYTHK2odJkoM6/view

A licensed plate detector was used to detect license plates. The model was trained with Yolov8 using
Link: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4

Following this step-by-step tutorial on how to train an object detector with Yolov8 on your custom data.
Link: https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide

Thank you original code from the youtube channel 
"Computer vision engineer" 
Link: https://youtu.be/fyJB1t0o0ms

---------------------------------------------------
*** Please Read requirements.txt file to setting environment ***

The process of program:
1. Import VDO => sample.mp4
2. Detect License Plate
3. Process License Plate
4. Write into CSV File => test.csv 
5. Cleaning CSV File => run add_missing_data using test.csv  and convert file to be test_interpolated.csv
6. Finding Index ID => run visualize.py to read_interpolated to visual index number plate
7. Combine CSV&VDO Process License Plate => out.mp4
8. Display Result 

---------------------------------------------------
Details of code as below:
0.	Import Library
1.	Load Models
2.	Load Video
3.	Read Frames
4.	Detect Vehicles
5.	Track Vehicles
6.	Detect License Plates
7.	Assign License Plate to car
8.	Crop License Plate
9.	Process License Plate
10.	Read License Plate Number
11.	Write Results
