**Vehicle Detection Project**

**My code: P5-hkawanishi-2.ipynb**

**My Project Video: project_output.mp4**


The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_image_example.jpg
[image2]: ./output_images/noncar_image_example.jpg
[image3]: ./output_images/car_hog_image_example.jpg
[image4]: ./output_images/noncar_hog_image_example.jpg
[image5]: ./output_images/car_image_with_boxes1.jpg
[image6]: ./output_images/heat_image1.jpg
[video1]: ./project_output.mp4


###Histogram of Oriented Gradients (HOG)

####1. 

The HOG feature (`get_hog_features()` function) is contained in the third cell.  This function calls "hog" function.  When `vis = True` argument is passed, it will return a hog image.  

`get_hog_features()` function is called by `extract_features()` (in cell 3) and `single_img_features()` (in cell 4).  `single_img_features()` is written mainly for a test so that I can verify a code by using a single image window and it was because I was following the lesson and this function was included.  Looking back, I could probably use only `extract_features()`.

I started by reading images in the `vehicles` and `non-vehicle` from the given zip files.  
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car image example][image1]
![non-car image example][image2]

I added color features and color histgram (`bin_spatial()` and `color_hist()` in cell 3).  I spent a lot of time deciding the parameters such as (`orientations`, `color_space`, `pix_per_cell`, and `hog_channels`).
I didn't use a clever statistic method to select these parameters. These were selected by trial-and-errors. The decision factors are based on (1) the accuracy from the training and (2) how the model detects the vehicles.  I will discuss further on these two points later.

Here is an example using the `YCrCb` color space and HOG parameters of `orientation=9`, `pix_per_cell=8`, `spatial_size=(16,16)`.

![car hog image example][image3]
![noncar hog image example][image4]

####2. How I seleted final HOG parameters.

As I wrote above, I selected the parameters by trial-and-errors. I will discuss more about how I trained a classifier in the following section.  But `YCrCb` for color space and HOG parameter of `orientation=9` were selected after changing the parameters and check the accuracy from the training.

For color space, I changed the parameter (e.g., `HSV`, `RGB`, etc.) and ran using 1000 random samples (each for both cars and non-cars) for three times.  From my quick test, I got slightly higher training accuracy using `HSV` and `YCrCb`.  I tried both and eventually decided to use `YCrCb` because it gave me a fewer false positive when detecting the vehicle. 

For `orientation`, I tried 6 and 9, and I got slighly better accuracy for the value 9 so I kept it that.

Also, I noticed that the vehicle detections improved a lot using `hog_channels = ALL` instead of using 0, 1 or 2.  


####3. How I trained a classifier using your selected HOG features (and color features if you used them).

This is done in cell 6 of my code.
I used 3000 random samples for cars and and for noncars.  I tried using more (~6000) but this slows my computer down and didn't give much improvement in the vehicel detections so I decided to settle with 3000.  The list of the image files are already shuffled in cell 1 so I ust used the first 3000 samples.
I split 0.2 for test and the rest for the training.  

I used all three features (spatial, color histogram, and HOG).  I tried just HOG but it gave me more false positive.  

I used a linear SVC.  I didn't play arount other method since I believed I got a decent accuracy.  

The accuracy result is as follows:

92.3 Seconds to extract features...
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6108
2.34 Seconds to train SVC...
Test Accuracy of SVC =  0.9875
My SVC predicts:  [ 1.  1.  1.  1.  1.  1.  0.  0.  0.  1.]
For these 10 labels:  [ 1.  1.  1.  1.  1.  1.  0.  0.  0.  1.]
0.00348 Seconds to predict 10 labels with SVC

###Sliding Window Search

####1. How I implemented a sliding window search.  

(1) First implemented `slide_window()` function which is included in cell 3 of my code. I pretty much followed our lesson, so first I used this function and detemine if I can have confidence in my parameter setting.  Sliding windows search is included cell 7.  From this test, I could roughly see where I could crop the original image, and see the features I applied gave me decent results.
(2) Next I implemented "HOG Sub-Sampling Window search" (It took a long time to do this...).  The function is called `find_cars()` and I included it in cell 10.  
I used test images to see this works (cell 11).  To prevent more false positive, I removed the sky and the dashboard portion; I used between the area `y = 400` and `y=656`.  (note that I went back to tweaked the color space and the HOG parameters based on these visual results).  
I found that `scale = 2` works better than `scale = 1` for the example.  I

####2. Show some examples of test images to demonstrate how your pipeline is working.  

I first tried HSV with 1-channel but I didn't get a good result. (a large rectangle window across a whole image...) In the end, I decided YCrCb and `hog_channels = ALL` parameters are good choice.  I also found that `scale = 2` give me a better result than 1 or even 1.5.  

Here are some example images:

t seems to give more false positive with smaller scale.  The example result with this "HOG sub-sampling window search" is shown below:

![HOG sub-sampling window search result][image5]

To avoid the overlapping detection, I implemented the heat map in `find_cars()` (cell 11).  The heat map result for the above image is shown below:

![heat map][image6]
---

### Video Implementation

####1. Video Output.
Just like my previous project, I imported VideoFileClip from moviepy.editor (cell 13) and call my `find_cars()` (cell 11) function from `process_image()` function (cell 14).  
My video result is:
[video1]: ./project_output.mp4

Sometimes the model picked up the vehicles on the opposite lanes on the left.  I should probably crop the left side of the image.  Also, when the vehicle goes farther (gets smaller), it is not detecting any more.  If I have more time, I would like to play around the paremeters more to get a better result.  


####2. Describe how I implemented to reduce false positives. 

To avoid multiple detections and reduce false positives, I implemented heat-map scheme inside `find_cars()` function (cell 11).  I zeroed out pixels with negative values in `apply_threshold()` (cell 10).  I imported `label` from `scipy.ndimage.measrements`  (cell 10) and put a bounding box around the labeled regions (`draw_labeled_dboxes()` in cell 10).  

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
(Heat map result example is shown in above, section 2).  


###Discussion

####1. Discuss any problems / issues you faced in your implementation of this project.  

* I had a hard time implementing HOG sub-sampling method.  I think it was because it was hard for me to debug compare with sliding windows method.  It seemed HOG sub-sampling method is more sensitive to parameter changes.  
* My pipeline doesn't seem to work too well when the vehicle moves further.  I need to tweak the parameters more.  
* I mentioned somewhere above, but I could probably crop the left-side of the image where the traffic goes to the opposite direction.  Though, this might not work well, if the driving vehicle goes to a winding road. 
* I didn't check to all the images see what sort of images are included.  But if there are only "cars" in the image data under vehicle folder, I am not sure if it will detect a large truck or a semi-trailer or motorcycle.  More traffic data are needed.


