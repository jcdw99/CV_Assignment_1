# Correction and Feature Detection

You will notice that this project was entirely implemented in python. Mandatory packages include Numpy, Pillow, MatPlotLib, and OpenCV. If you need to install one of these packages, use pip. An example of numpy's installation is as follows:

```bash
pip install numpy
```

Once you have installed the required packages, you are ready to run the code. All questions consist of fairly modular code that is highly commented, feel free to consult the function comments for a description of the functions use. Otherwise, A breakdown of the usage of each question is provided below.

## Question 1
Question 1 does not contain variables that should be changed to yield different results. If you run the program you will make a call to the apply_all() function. This function will write output images to the directory Resources/Output/Monkey, and will display the histograms on screen. If you wish to change the image, add the desired image to the Resources/ directory, and change the path to img at the top of the program.

## Question 2
If you run this program, all the relevant code is executed, though the images are not shown. You will notice at the bottom of the program there is a collection of Image objects which are assigned. If you would like to view any of these objects, simply add the line <Image_name>.show(). This will show the image. 

An example of showing the image, ''corrected'' is commented out, feel free to uncomment it and view the results. If you wish to change the corruption threshold 'd', or the median mask size, you will notice on line 56 and 58, that these variables can be changed accordingly. If you wish to change the image, add the desired image to the Resources/ directory, and change the path to img at the top of the program.

## Question 3
This code contains a variable towards the top of the program named 'k'. This variable determines how many edge images should be added to the original image. If you want a sharper result, increase K.  The function show_results() runs the entire program. This function is called at the bottom of the program, so simply run the python file, and the results will be automatically shown on screen, for various values k. (1 to 6).

## Question 4
At the top of the code is a variable named size_Factor, which determines the multiple of pixels in the output image, as a proportion of pixels in the input image. For example, a size_factor of 2, implies the output image will contain double the pixels of the input image. Another variable towards the top of the code is called Nearest_neighbor. When this variable is set to True, Nearest_Neighbor interpolation is run, otherwise, Bilinear interpolation is run. Finally, the path to the input image is read at the top of the code as well. The method which actually implements the mapping is titled good_apply_transform(). It was given this name because I have also implemented a bad_apply_transform() which does a forward mapping instead of a backwards mapping. This forward mapping results in some pixels not being mapped to in the output image. The results of this code are written to Resources/Output/Resizes. Navigate to that directory to view the results.

## Question 5
At the top of the code, the images are read from the root directory. There are 3 important functions. The first function, get_point_data(), computes the key points and descriptors of a provided input image. The second function, get_matches(), runs a flannBasedMatcher to determine keypoint matches between the images. The final function, draw_lines(), visualizes the results. Draw_lines() assumes the other functions have been run. It makes more sense to run question6's code instead of question5, as question6 is more modular, and allows for higher customisation.

## Question 6
Question 6 combines code from Questions 4 and Questions 5. The main function here is called run_simul(). You will notice in this function that there is a variable set to False, this variable will not run the full test procedure, but rather simply read static data, corresponding to the output of the testing procedure. Because this process is not stochastic, we can expect the same results always. If you would like to run the simulation yourself, simply change this variable to True, and the simulation is run.

If you wish to run (and view) certain resize image pairs yourself, including the plots, please navigate to the do_iteration() function, and simply uncomment the results.show() found in this method. When you subsequently call do_iteration(), be sure to pass in the desired input image, and resize factor. 

## Contributing
All the code was written by myself, unless a library was explicitly credited, in which case the library was used. I did not share code with peers, nor get code from my peers.
