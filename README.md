# Project Description
This Project is divided into two parts:

**1. Straight Lane Detection**: In this part, the goal is to do simple Lane Detection to mimic Lane Departure Warning systems used in Self Driving Cars. You are provided with a video sequence, taken from a car. Your task is to design an algorithm to detect lanes on the road, and classify them as dashed and solid lines. For classification of the line type, you have to use different colors. Use green for solid and red for dashed.

**2. Curve Lane Detection and Turn Prediction**: In this part, the goal is to detect the curved lanes and predict the turn depending on the curvature: either left or right turn. The dataset provided has a yellow line and a while line. Your task is to design an algorithm to detect these lanes on the road, and predict the turn. When your lane detection system loses track(cannot find lane lines), you must manage to use the past history of the lanes from the previous image frames to extrapolate the lines for the current frame.

# Dataset

[Dataset for Problem 1](https://drive.google.com/drive/folders/1vW5Vp2h6IEHvEeDrsfjqwJT7ZBwzXXf-?usp=sharing)\
[Dataset for Problem 2](https://drive.google.com/file/d/1e7Xy_FiP64alDXoIw2Pxrh2_cVfTQYT-/view?usp=sharing)

# Pipeline

## Straight Lane Detection
**1. Solid Lane Detection:**
  - Apply mask to detect region of interest(part of the image containing lane to be detected).
  - Apply [Hough Transform](https://en.wikipedia.org/wiki/Hough_transform) with minimum length value(decided by tuning) high enough to detect only solid lane.
  - Find the mean of all the start and end points.
  - Find the slope mean value calculated in step 2.
  - Draw a single line on the image with calculated slope.
  
 **2. Dashed Lane Detection:**
  - Apply mask to detect region of interest(part of the image containing lane to be detected).
  - Apply Hough transform to detect all the line in the masked image.
  - Find the slope of all the lines and remove those lines which is positive slope if solid lane has positive slope and negative slope if solid line has negative slope.
  - Find the mean of all the start and end points in the remaining lines.
  - Find the slope with mean value calculated in step 3.
  - Draw a single line with calculated slope.
  
 ## 



https://user-images.githubusercontent.com/90370308/216849619-58321f50-1499-43d4-9470-595fa1d5e3b6.mp4

![Question2_output_AdobeExpress](https://user-images.githubusercontent.com/90370308/216849852-71620780-1579-40fd-a5c7-514c8e85c579.gif)

https://user-images.githubusercontent.com/90370308/216849814-b87edaca-4676-441b-9dd6-b929968f15ab.mp4

![Question3_output_AdobeExpress](https://user-images.githubusercontent.com/90370308/216849943-2f6d82cd-ea7e-4c5b-87f7-72b985e5ad3e.gif)
