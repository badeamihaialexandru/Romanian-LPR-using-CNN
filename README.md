  This project takes as input an image and outputs a string containing the number plate in that image if it does exist. As any other LPR
 it has 3 main steps:
  1. Number plate segmentation
  2. Character segmentation 
  3. Character recognition 
  
  For the segmentation of both number plate and characters, I used a classic method based on CCA algorithm, which is explained very well 
  here: 
  https://www.youtube.com/watch?v=ticZclUYy88&t=84s
  
  The recognition was done using a Convolutional Neural Network. For its training I selected 1000 images from an online Data Set. I added
  the mnist files in the "Training Data" directory. I did the conversion of the pictures from jpeg to mnist idx using this program: 
  https://github.com/nyanp/mnisten 
  
  The resulting acuracy was 96% and the time it needs for output is short.
  
  
  
