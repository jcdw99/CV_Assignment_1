from PIL import Image
import numpy as np

img = Image.open("Resources/hippo.jpeg")

""" Corrupts a provided colour channel by the percentage specified as the argument """
def corrupt_band(pic, d=0.2):
    data = np.array(pic)
    raw_data = np.zeros((data.shape))

    for row in range(len(raw_data)):
        for col in range(len(raw_data[row])):
            if np.random.random_sample() < d:
                # check if we should mutate
                if np.random.random_sample() < 0.5:
                    # white
                    raw_data[row][col] = 255
                else:
                    # black
                    raw_data[row][col] = 0
            else :
                raw_data[row][col] = data[row][col]
    return Image.fromarray(raw_data)

""" Merges colour channels into a single RGB image"""
def combine_bands(rdata, gdata, bdata):
    rdata = rdata.convert("L")
    gdata = gdata.convert("L")
    bdata = bdata.convert("L")
    return Image.merge("RGB", (rdata, gdata, bdata))

""" Performs Median filteration with 0's subsituted for out of range requests"""
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
                    
            median = np.median(np.array(cells))
            new_data[row][col] = median

    return Image.fromarray(new_data)


raw_bands = [*img.split()]
corrupt_bands = [corrupt_band(i, d=0.6) for i in raw_bands]
corrupt_combined = combine_bands(*corrupt_bands)
median_corrected = [convolve(i, np.ones((11,11))) for i in corrupt_bands]
corrected = combine_bands(*median_corrected)
""" If you wish to show any images, simply add <img_name>.show() """
# corrected.show()
