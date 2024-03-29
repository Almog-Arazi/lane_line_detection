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

## Results



### Day Result
https://drive.google.com/file/d/1XmTtlc9z_W5AmOKVsSNyUQeJPg3Zqh9J/view?usp=sharing
### Night-Time Lane Detection Result
![Night-Time Lane Detection Result](https://s6.ezgif.com/tmp/ezgif-6-2f3c82a481.gif)

### Crosswalk Detection Result
<iframe width="560" height="315" src="https://www.youtube.com/embed/pUXdubbXLzc?si=4nmi3cVjFLzb3V2Q" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
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
