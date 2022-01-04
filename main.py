from NeuralNetworkHandler import NeuralNetworkHandler
from ImageProcessor import ImageProcessor
import os


dir = os.path.dirname(os.path.realpath(__file__))
nn = NeuralNetworkHandler()
image = ImageProcessor(dir+"/TestImages/"+"0_1_0.png")

temp = nn.evaluate_image(image_data=image.image_data.data)
print(temp)
