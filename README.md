# Lane Detection for Autonomous Vehicles

This project focuses on implementing a lane detection system for autonomous vehicles using computer vision techniques. The project encompasses various aspects of image processing and analysis, including lane marking detection, night-time adaptation, and crosswalk detection. The algorithm aims to accurately identify lane markings under different conditions and enhance the safety and reliability of autonomous driving systems.

## Core Project

### Video Selection and Processing
- Choose a dashcam video featuring a car driving on a highway.
- Extract a segment of at least 20 seconds from the video.
- Consider factors such as lighting conditions, lane visibility, and road type.

### Lane Marking Detection and Visualization
- Annotate each frame of the video with accurate lane markings.
- Include a demonstration of a lane change with appropriate markings.
- Implement strategies for cropping, color thresholding, and line verification.

### Example Frame Analysis
- Provide an in-depth analysis of an example frame.
- Illustrate the approach to identifying and marking lane lines.
- Address complexities like varying light conditions or nearby traffic.

## Night-Time Lane Detection

### Objective
Adapt the lane detection algorithm for low-light conditions.
Implement image enhancement techniques to improve lane visibility.
Ensure high accuracy comparable to daytime performance.

## Crosswalk Detection

### Objective
Enhance the lane detection system with crosswalk detection capability.
Implement algorithms to recognize crosswalk patterns and lines.
Differentiate crosswalks from other road markings for safety.

## The Architecture of the Algorithm: Step-by-Step Process

![דגשגשג](https://github.com/user-attachments/assets/78df63a7-db15-45f2-aa6e-7d5a922c7962)


## Results

### Day Result


<a href="https://www.youtube.com/embed/aVcRgJhPpVo?si=OBlkvJ-054Qv1GUr">
  <img src="https://img.youtube.com/vi/aVcRgJhPpVo/maxresdefault.jpg" alt="Watch the Video" width="600" height="400">
</a>

### [Watch the Video Here](https://www.youtube.com/embed/aVcRgJhPpVo?si=OBlkvJ-054Qv1GUr)


### Night-Time Lane Detection Result



<a href="https://www.youtube.com/embed/rlmJ9cQNHYw?si=WkvmkkUjClX2JpNZ">
  <img src="https://img.youtube.com/vi/rlmJ9cQNHYw/maxresdefault.jpg" alt="Watch the Video" width="600" height="400">
</a>

### [Watch the Video Here](https://www.youtube.com/embed/rlmJ9cQNHYw?si=WkvmkkUjClX2JpNZ)


### Crosswalk Detection Result


<a href="https://www.youtube.com/embed/pUXdubbXLzc?si=BMTVP0C6eUFqbnXu">
  <img src="https://img.youtube.com/vi/pUXdubbXLzc/maxresdefault.jpg" alt="Watch the Video" width="600" height="400">
</a>

### [Watch the Video Here](https://www.youtube.com/embed/pUXdubbXLzc?si=BMTVP0C6eUFqbnXu)


## Python Code Overview

### Day and Night Algorithm
- Implements lane detection for both daytime and nighttime conditions.
- Utilizes Canny edge detection, Hough transform, and image manipulation.
- Includes functions for adjusting brightness and contrast.

### Crosswalk Algorithm
- Detects crosswalks alongside lane markings in video footage.
- Integrates contour detection and pattern recognition techniques.
- Enhances safety features for comprehensive lane detection.

**Note:** Ensure appropriate video inputs and parameters for optimal performance of the algorithms.
