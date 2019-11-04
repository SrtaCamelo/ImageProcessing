import os
import cv2
import imutils
import pandas as pd
from skimage.feature import hog

main_path = "../Imagens_pulmão/"
main_path = "C:/Users/SrtaCamelo/Documents/2019_2/ProcessamentoDeImagem/ImageProcessing/Imagens_pulmao/"
lungs = ["nan_nodule","nodule"]
number = {"nan_nodule": 0, "nodule":1}


"""
Threshold aplication
"""
def preprocess_limiar(img):
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    return th1
"""
Median Blur aplication
"""

def preprocess_median(img):
    median = cv2.medianBlur(img, 27)

    return median

"""
Gaussian Blur aplication
"""

def preprocess_gaussian(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    return blur
"""
Auxilliary Function to display image.
"""

def show(type,img):
    # print(img)
    cv2.imshow(type, img)
    cv2.waitKey()


def preprocess(flag):
    file_list = []
    file_name = ""
    for lung in lungs:
        print(lung)
        path = main_path+ lung
        files = os.listdir(path)

        for file in files:
            dirr = path + "/" + file
            # print(dirr)
            # dirr = "C:\\Users\\SrtaCamelo\\Desktop\\diagx11.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.png"
            # dirr = "C:\\Users\\SrtaCamelo\\Documents\\2019_2\\ProcessamentoDeImagem\\ImageProcessing\\Imagens_pulmao\\nan_nodule\\diagx11.3.6.1.4.1.14519.5.2.1.6279.6001.100398138793540579077826395208.png"
            img = cv2.imread(dirr)
            # show(lung,img)
            if(flag == 1):
                new_img = preprocess_limiar(img)
                # show(type,new_img)
                file_name = "limiarizado_nodules.csv"

            elif(flag == 2):
                # print("Gaussian Blur")
                new_img = preprocess_gaussian(img)
                # show(type, new_img)
                file_name = "gaussian_nodules.csv"

            else:
                # print("Median Blur")
                new_img = preprocess_median(img)
                # show(type, new_img)
                file_name = "median_nodules.csv"

            resized_image = imutils.resize(new_img, width=80, height=80)
            fd, hog_image = hog(resized_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                                visualize=True,
                                feature_vector=True)


            fd = fd.tolist()
            fd.append(number[lung])
            file_list.append(fd)
    df = pd.DataFrame.from_records(file_list)
    df.to_csv(file_name)

"""
print("Trheshold")
preprocess(1)
print("Gaussian Blur")
preprocess(2)
"""
print("Median Blur")
preprocess(3)