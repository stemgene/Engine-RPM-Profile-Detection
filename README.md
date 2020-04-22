# Engine-RPM-Profile-Detection

## 1. Introduction

**[Sponsor](https://www.vnomicscorp.com/)**

**Main Purpose**: To find 2 representative indicators, HighRPM and MaxRPM, with certain statistical significance.

## 2. [Data Inspection](https://github.com/stemgene/Engine-RPM-Profile-Detection/blob/master/Data_Inspection.ipynb)

## 3. Approaches

- 3.1 Summarization Approach

  - Analyze each single trip of a specific platform.

  - Summarize the results from all trips of a specific platform to predict RPM profile of the platform.

- 3.2 Regression Approach

  - Find the inflection point of the torque from maximum value.
  
<p align="center">
<img src="https://github.com/stemgene/Engine-RPM-Profile-Detection/blob/master/img/01.png" alt="drawing" width="500"/>
</p>  

### 3.2.1 Main steps

- Step 1: Find the contour curve of data

<p align="center">
<img src="https://github.com/stemgene/Engine-RPM-Profile-Detection/blob/master/img/02.png" alt="drawing" width="500"/>
</p>  

- Step 2: Introduce linear regression

  - Fitting N lines of linear regression
  
  - Horizontal line of the maximum of torque
  
  - N intersections among these two types of lines

<p align="center">
<img src="https://github.com/stemgene/Engine-RPM-Profile-Detection/blob/master/img/03.png" alt="drawing" width="500"/>
</p>  

- Step 3: Training the model and determine the optimized intersection.

  - Ground truth: The theoretical value of HighPRM in handbook
  
  - Supervised learning to determine N
  
  - The intersection of two lines will be the predicted value.
  
<p align="center">
<img src="https://github.com/stemgene/Engine-RPM-Profile-Detection/blob/master/img/04.png" alt="drawing" width="500"/>
</p>  

- Step 4: Statistical significance

<p align="center">
<img src="https://github.com/stemgene/Engine-RPM-Profile-Detection/blob/master/img/05.png" alt="drawing" width="500"/>
</p>  
