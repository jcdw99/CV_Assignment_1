from PIL import Image, ImageOps
import numpy as np


img = Image.open("Resources/tiger.jpeg")
img = ImageOps.grayscale(img)
k = 5

"""  Blurs an image using a standard mean filter """
def convolve(channel, mask):

    raw_data = np.array(channel)
    new_data = np.zeros((raw_data.shape))

    maskWidth = int((len(mask) - 1 ) / 2)
    for row in range(len(raw_data)):
        for col in range(len(raw_data[row])):
            cells = []
            for i in range(-maskWidth, maskWidth, 1):
                for j in range(-maskWidth, maskWidth, 1):
                    try:
                        cells.append(raw_data[row-i][col-j])
                    except IndexError:
                        # out of bounds, pad in a zero
                        cells.append(0)
                    
            mean = np.mean(np.array(cells))
            new_data[row][col] = mean

    return Image.fromarray(new_data)

""" Performs unsharp masking of provided intensity, on provided image"""
def perform_unsharp(pic, k):
    blurry_pic = convolve(pic, np.ones((5,5)))
    sub_pic = Image.fromarray(np.array(blurry_pic) - np.array(pic))
    mult_sub_pic = Image.fromarray(np.array(sub_pic) * k)
    result = Image.fromarray(np.array(pic) + np.array(mult_sub_pic))
    return result

""" Runs everything """
def show_results():
    results = [perform_unsharp(img.copy(), i) for i in range(1,6)]
    for result in results:
        result.show()


show_results()
