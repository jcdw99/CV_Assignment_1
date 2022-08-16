from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#This is the Image we will consider
img = ImageOps.grayscale(Image.open("Resources/monkey.png"))

"""  This is the Transformation Function based upon hyperbolic tangent """
def T(s, c3=1.5):
    # tanh is like a sigmoid function, maps a positive x to a value y between 0,1
    # tanh is always between -1 and 1, for all its domain, and strictly positive for positive domain
    k=255
    c4 = k/2
    c1 = (-c4) / np.tanh(-c3)
    c2 = c3 / (c4)

    return c1 * np.tanh(c2 * s - c3) + c4


"""
Utility functions are below. The Purpose of these functions are to perform transformations and visualizations on
various images at the same time, and display the results using pyplot. 
"""

"""  Expects a list corresponding to post-transformation vectors. Plots this list against standard [0-255] axis   """
def plot_transform(transform):
    plt.plot(list(range(256)), transform, list(range(256)), list(range(256)))    

"""  apply a transformation give in list form, where each element of the list represents where the corresponding index gets transformed to"""
def apply_transformation(pic, trans):
    result = pic.point(lambda pix: int(trans[pix]))
    return result

""" This function plots the transform family that I have designed, It plots 11 curves, as well as the line y=x. Used for report/visualization """
def plot_trasnform_family():
    for i in range(0,11):
        c3 = 1+(i/2)
        k=255
        c4 = k/2
        c1 = (-c4) / np.tanh(-c3)
        c2 = c3 / (c4)
        lab = str('c1=' + str(c1)[:7] + ", c2=" + str(c2)[:7] + ', c3=' + str(1+i/2)[:7])
        # plt.plot(T(np.arange(0,256), 1.0 + (i/2)), label='c3='+str(c3))

        plt.plot(T(np.arange(0,256), 1.0 + (i/2)), label=lab)

        
    plt.plot(list(range(256)))
    plt.title("Contrast Stretching Based on Hyperbolic Tangent")
    plt.legend()
    plt.show()

""" Gets a gamma transform corresponding to a value of gamma, provided as argument """
def get_gamma_trans(gam):
    return [int(255*(i/255)**gam) for i in range(256)]

""" Plots a family of gamma transforms, for visualization purposes and the report.. """
def draw_onion():
    plt.xlim([0,255])
    plt.ylim([0,255])
    plt.plot(list(range(256)), get_gamma_trans(.04), label="g=0.04")
    plt.plot(list(range(256)), get_gamma_trans(.1), label="g=0.1")
    plt.plot(list(range(256)), get_gamma_trans(.2), label="g=0.2")
    plt.plot(list(range(256)), get_gamma_trans(.4), label="g=0.4")
    plt.plot(list(range(256)), get_gamma_trans(.67), label="g=0.67")
    plt.plot(list(range(256)), list(range(256)), label="y=x", linewidth=4, color='black')
    plt.plot(list(range(256)), get_gamma_trans(1.5), label="g=1.5")
    plt.plot(list(range(256)), get_gamma_trans(2.5), label="g=2.5")
    plt.plot(list(range(256)), get_gamma_trans(5), label="g=5")
    plt.plot(list(range(256)), get_gamma_trans(10), label="g=10")
    plt.plot(list(range(256)), get_gamma_trans(25), label="g=25")
    plt.legend()
    plt.xlabel("Input Pixel Intensity")
    plt.ylabel("Ouput Pixel Intensity")
    plt.show()

""" Plot the tanh(x) function along with a shifted version, for purposes of report """
def plot_tanh():
    in_array = np.linspace(-np.pi, np.pi, 100)
    out_array = np.tanh(in_array)
    out_array2 = np.tanh(in_array) + 1
    plt.plot(in_array, out_array, color = 'red', marker = ".", label='tanh(x)')
    plt.plot(in_array, out_array2, color = 'blue', marker = ".", label='tanh(x) + 1')
    plt.legend()
    plt.title("The Hyperbolic Tangent Function")
    plt.show()


""" Plot the desired transformation, for the purpose of the report """
def plot_desired():
    plt.plot(np.linspace(0, 256, 100), T(np.linspace(0, 256, 100), 3), color='blue', marker = ".")
    plt.plot(list(range(256)), list(range(256)), label="y=x")
    plt.legend()
    plt.title("The Desired Transformation")
    plt.show()

""" Apply each transformation specified in the family of curves, to an input image """
def apply_all():
    img.save("Output/Monkey/monk_orig.png")
    hst=img.histogram()
    plt.figure(0)             # plots a figure to display RED Histogram
    for i in range(0, 256):
        plt.bar(i, hst[i],color='grey',alpha=0.8)
    plt.savefig("Output/Monkey/hist_monk_orig.png")
    plt.show()
    for i in range(0,11):
        c3 = 1+(i/2)
        output = apply_transformation(img.copy(), T(np.arange(0,256), c3))
        output.save("Output/Monkey/c3=" + str(c3) + ".png")
     
        hst=output.histogram()
        plt.figure(0)             # plots a figure to display RED Histogram
        for i in range(0, 256):
            plt.bar(i, hst[i],color='grey',alpha=0.8)
        plt.savefig("Output/Monkey/hist_c3=" + str(c3) + ".png")
        plt.show()
            
apply_all()
data = np.array(img)
avg = np.average(data)
output = apply_transformation(img, T(np.arange(0,256), 2))
img.show()
output.show()
 