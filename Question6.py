from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

pic1 = cv.imread('Resources/rushmore.jpeg')
pic1 = cv.cvtColor(pic1,cv.COLOR_BGR2GRAY)


def get_point_data(img):

    image8bit = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    gray = cv.cvtColor(image8bit, cv.COLOR_RGB2BGR)
    sift = cv.SIFT_create()
    kp, desc = sift.detectAndCompute(gray,None)
    return (kp, desc)


def get_matches(points1, points2):
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(points1, points2, 2)
    good_matches = []

    for m, n in matches:
        if m.distance < 0.3 * n.distance:
            good_matches.append(m)
    return good_matches

def draw_lines(pic1, kp1, kp2):
    draw = ImageDraw.Draw(pic1)
    for index in range(len(kp1)):
        #extremely long matches are usually errors, remove
        srcx, srcy = kp1[index]
        destx,desty = kp2[index]
        draw.ellipse((srcx-3, srcy-3, srcx+3, srcy+3))
        draw.line([(srcx, srcy), (destx, desty)], fill='red', width=1)

def get_transformation_matrix(factor):
    return np.identity(2) * np.sqrt(factor)

def invert_matrix(factor):
    matrix = get_transformation_matrix(factor)
    return matrix / factor

def transform_vector(matrix, vec):
    return matrix.dot(vec)

def is_good_mapping(factor, kp_coords_src, kp_coords_dest):
    back_mat = invert_matrix(factor)
    kp_coords_src = np.array([kp_coords_src[0], kp_coords_src[1]])
    kp_coords_dest = np.array([kp_coords_dest[0], kp_coords_dest[1]])

    should_be = transform_vector(back_mat, kp_coords_dest)
    dist = np.linalg.norm(should_be - kp_coords_src) 
    return dist < 2
     

def good_apply_transformation(pic, size_factor):
    blank = np.zeros((int(pic.size[1] * np.sqrt(size_factor)), int(pic.size[0] * np.sqrt(size_factor)), 3)).astype(np.uint8)
    matrix = invert_matrix(size_factor)
    orig_data = np.array(pic)
    rows = len(orig_data)
    cols = len(orig_data[0])
    for y in range(len(blank)):
        for x in range(len(blank[y])):
            dest = transform_vector(matrix, np.array([y, x]))
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


def do_iteration(factor, pic1):
    pic2_orig = good_apply_transformation(pic1, factor)
    pic2 = np.float32(np.array(pic2_orig))
    pic1_kp, pic1_desc = get_point_data(np.float32(np.array(pic1)))
    pic2_kp, pic2_desc = get_point_data(pic2)

    matches = get_matches(pic1_desc, pic2_desc)

    list_kp1 = [pic1_kp[mat.queryIdx].pt for mat in matches]
    list_kp2 = [pic2_kp[mat.trainIdx].pt for mat in matches]

    duplicate = pic2_orig.copy()
    results = draw_lines(duplicate, list_kp2, list_kp1)
    # results.show()
    good_maps = 0
    bad_maps = 0
    for index in range(len(list_kp2)):
        correct = is_good_mapping(factor, list_kp1[index], list_kp2[index])
        if correct:
            good_maps = good_maps + 1
        else:
            bad_maps = bad_maps + 1
    

    print(str(good_maps / (good_maps + bad_maps) * 100)[:4] + "% accurate for factor: " + str(factor))
    return ((good_maps / (good_maps + bad_maps)) * 100), len(matches)


def run_simul():
    # The Mean match length is 618.3333333333334
    data = []
    run_simul = False
    if run_simul:
        match_len = []
        for i in np.arange(0.1, 3.1, 0.1):
            results = do_iteration(i,  Image.open("Resources/rushmore.jpeg"))
            data.append(results[0])
            match_len.append(results[1])
        print(data)
        print(np.mean(match_len))
    else:
        prev = [91.22807017543859, 93.7062937062937, 96.875, 99.43342776203966, 97.77777777777777, 99.40476190476191,\
                97.94721407624634, 98.12206572769952, 98.88579387186628, 100.0, 99.47826086956522, 98.71382636655949,\
                99.08256880733946, 98.77149877149877, 99.44341372912801, 99.85074626865672, 99.85119047619048, 100.0,\
                99.0909090909091, 99.13194444444444, 99.25093632958801, 99.15492957746478, 99.33510638297872, 99.8663101604278,\
                99.8812351543943, 100.0, 99.76635514018692, 99.63369963369964, 99.45872801082544, 99.38271604938271]
        data = prev

    plt.plot(np.arange(0.1, 3.1, 0.1), data)
    plt.ylim(0,105)
    plt.ylabel("% Accuracy")
    plt.xlabel("Pixel Count Multiplier")
    plt.show()

run_simul()