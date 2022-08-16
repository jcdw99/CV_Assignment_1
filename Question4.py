from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

size_factor = 8
nearestNeighbor = True
img = Image.open("Resources/waterfall.jpeg")
blank = np.zeros((int(img.size[1] * np.sqrt(size_factor)), int(img.size[0] * np.sqrt(size_factor)), 3)).astype(np.uint8)

def get_transformation_matrix():
    return np.identity(2) * np.sqrt(size_factor)

# this method leads to black cells, so we should do inverse mappings first..
def bad_apply_transformation():
    matrix = get_transformation_matrix()
    data = np.array(img)
    for row in range(len(data)):
        for col in range(len(data[row])):
            destination = transform_vector(matrix, np.array([row, col]))
            blank[destination[0]][destination[1]] = np.array(data[row][col]).astype(np.uint8)
    return Image.fromarray(blank)  

def good_apply_transformation(mode=nearestNeighbor):
    matrix = invert_matrix()
    orig_data = np.array(img)
    rows = len(orig_data)
    cols = len(orig_data[0])

    for y in range(len(blank)):
        for x in range(len(blank[y])):
            dest = transform_vector(matrix, np.array([y, x]))
            if mode:
                dest = (np.round(dest)).astype(int)
                dest[0] = dest[0] if dest[0] < rows else rows - 1
                dest[1] = dest[1] if dest[1] < cols else cols - 1
                blank[y][x] = np.array(orig_data[dest[0]][dest[1]]).astype(np.uint8)
            else:
                xf = int(dest[1])
                yf = int(dest[0])
                xc = (xf + 1 if xf + 1 < cols else cols - 1)
                yc = (yf + 1 if yf + 1 < rows else rows - 1)
                if xc == xf or yc == yf:
                    blank[y][x] = np.array(orig_data[yf][xf]).astype(np.uint8)
                else :   
                    blank[y][x] = (yc - dest[0]) * ((xc - dest[1]) * (orig_data[yf][xf]) + (dest[1] - xf) * (orig_data[yc][xf])) + \
                        (dest[0] - yf) * ((xc - dest[1]) * (orig_data[yf][xc]) + (dest[1] - xf) * (orig_data[yc][xc]))
    return Image.fromarray(blank)

def invert_matrix(matrix=get_transformation_matrix()):
    return matrix / size_factor

def transform_vector(matrix, vec):
    return matrix.dot(vec)
     
plt.imshow(good_apply_transformation(False))
plt.savefig("Output/Resizes/resized_" + str(size_factor) + "_bilat.png")
plt.imshow(good_apply_transformation(True))
plt.savefig("Output/Resizes/resized_" + str(size_factor) + "_near.png")
