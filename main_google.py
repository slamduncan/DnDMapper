import json
import os, random
import io
from google.cloud import vision
import datetime
from pathlib import Path
import pickle
from edge_image_processor import cannyEdgeDetector
from skimage.draw import polygon_perimeter, set_color
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw, ImageShow, ImageColor, ImageFilter
import numpy as np

from google.protobuf.json_format import MessageToDict


def detect_handwritten_ocr(path):

    # check if we have already done this file
    text_path = Path(Path(path).parent, Path(path).absolute().stem + '.txt')
    bin_path = Path(Path(path).parent, Path(path).absolute().stem + '.bin')
    #temp_txt = Path(text_path)
    if text_path.exists() and bin_path.exists():
        temp_img = Path(path)

        txt_mtime = datetime.datetime.fromtimestamp(text_path.stat().st_mtime)  # get last mod time
        img_mtime = datetime.datetime.fromtimestamp(temp_img.stat().st_mtime)  # get last mod time
        bin_mtime = datetime.datetime.fromtimestamp(bin_path.stat().st_mtime)  # get last mod time

        if txt_mtime >= img_mtime and bin_mtime >= img_mtime:
            print("Already did this image")
            with open(text_path) as f:
                data = f.readlines()
            with open(bin_path, 'rb') as config_dictionary_file:
                response_dict = pickle.load(config_dictionary_file)
            return response_dict
        else:
            print("text file out of date")

    else:
        """Detects handwritten characters in a local image.
    
        Args:
        path: The path to the local file.
        """
        from google.cloud import vision_v1p3beta1 as vision
        client = vision.ImageAnnotatorClient()

        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # Language hint codes for handwritten OCR:
        # en-t-i0-handwrit, mul-Latn-t-i0-handwrit
        # Note: Use only one language hint code per request for handwritten OCR.
        image_context = vision.ImageContext(
            language_hints=['en-t-i0-handwrit'])

        response = client.document_text_detection(image=image,
                                                  image_context=image_context)
        #response_crop = client.crop_hints(image=image, image_context=image_context)  #Need to figure out ratio for image?
        #hints = response.crop_hints_annotation.crop_hints

        try:
            with open(text_path, 'w') as fn:
                fn.write('Full Text: {}'.format(response.full_text_annotation))
            with open(bin_path, 'wb') as fn:
                response_dict = MessageToDict(response._pb)
                pickle.dump(response_dict, fn)
        except Exception as e:
            print("Issue writing date")
            print(e)

        print('Full Text: {}'.format(response.full_text_annotation.text))
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                print('\nBlock confidence: {}\n'.format(block.confidence))

                for paragraph in block.paragraphs:
                    print('Paragraph confidence: {}'.format(
                        paragraph.confidence))

                    for word in paragraph.words:
                        word_text = ''.join([
                            symbol.text for symbol in word.symbols
                        ])
                        print('Word text: {} (confidence: {})'.format(
                            word_text, word.confidence))

                        for symbol in word.symbols:
                            print('\tSymbol: {} (confidence: {})'.format(
                                symbol.text, symbol.confidence))
        return response_dict #response.full_text_annotation

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))


# Authentication to Google API
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'cloud_key/vison_key.json'
vision_client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.abspath("TestImages/"+"DnDMap.jpg")

data_dic = detect_handwritten_ocr(file_name)

text_data = data_dic['textAnnotations']
bounding_boxes = []
for detected in text_data:
    try:
        num = int(detected['description'])
        print('Num: ' + str(num))
        bounding_boxes = bounding_boxes + [(num, detected['boundingPoly'])]
    except Exception as e:
        print(e)

#with io.open("TestImages/"+"DnDMap.jpg", 'rb') as image_file:
asset_dict = {}
with Image.open("TestImages/"+"DnDMap.jpg") as img:
    img_edge = img.filter(ImageFilter.FIND_EDGES)
    img_edge.show()
    #content = image_file.read()
    #img = mpimg.imread("TestImages/"+"DnDMap.jpg")
    for num, bb in bounding_boxes:
        if not num in asset_dict.keys():
            fp = ""
            while 'png' not in fp:
                fp = random.choice(os.listdir("TestImages/Assets"))
            asset_dict[num] = Image.open("TestImages/Assets/"+fp)

        row, col = [], []
        for point in bb['vertices']:
            row = row + [point['x']]
            col = col + [point['y']]

        draw = ImageDraw.Draw(img)
        draw.rectangle([(row[0], col[0]), (row[2], col[2])], outline=(255, 0, 0), width=3)

        center_point = (row[2] - row[0])/2 + row[0], (col[2] - col[0])/2 + col[0]
        asset_w, asset_h = asset_dict[num].size
        center_point = (int(center_point[0] - asset_w/2), int(center_point[1] - asset_h/2))
        img.paste(asset_dict[num], center_point, asset_dict[num])
    ImageShow.show(img)

    bw_img = img.convert("L")
    CED = cannyEdgeDetector(imgs=[bw_img], sigma=3)
    edge_img = CED.detect()
    edge_img = Image.fromarray(edge_img[0])
    ImageShow.show(edge_img)
    a = 1


        #rr, cc = polygon_perimeter(row, col, shape=img.size, clip=False)
        #img_copy = img.copy()
        #a = np.asarray(img_copy)
        #a.setflags(write=1)
        #set_color(a, (rr, cc), (0, 0, 255))
        #plt.imshow(a)
        #plt.show()

# # Loads the image into memory
# with io.open(file_name, 'rb') as image_file:
#     content = image_file.read()
#
# image = vision.Image(content=content)
#
# # Performs label detection on the image file
# response = vision_client.label_detection(image=image)
#
# labels = response.label_annotations
#
# print('Labels:')
# for label in labels:
#     print(label.description)
#
# image = vision.Image()
#
# image_uri = 'https://upload.wikimedia.org/wikipedia/commons/b/bf/Mobile_phone_IMEI.jpg'
#
# image.source.image_uri = image_uri
