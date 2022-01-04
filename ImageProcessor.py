import cv2
from datatools.image_processing.image_data import *


class ImageProcessor():
    detected_num = None
    image = None
    image_data = None

    def __init__(self, path_to_image):
        try:
            self.image = cv2.imread(path_to_image)
            self.image_data = ImageData()
            self.image_data = self.image_data.loadImage(path_to_image)
        except:
            pass

    def identify_num(self):
        pass



    # id = id.loadImage("nntools/0_1_0.png")
    # out = nnc.evaluate(id.data)

    def evaluate_image(self, image_data):
        # image in shape of 28x28 png
        self.nnc.evaluate(image_data)

