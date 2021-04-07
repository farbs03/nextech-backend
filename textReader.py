import os, io
from google.cloud import vision
import pandas as pd
import numpy as np
import cv2
import base64
from imageio import imread

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"google_cloud_api.json"
client = vision.ImageAnnotatorClient()

def getTextFromImage(img_file):
    content = open(img_file, 'rb').read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    return response.text_annotations[0].description

def getImageDescription(img_file):
    content = open(img_file, 'rb').read()
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    desc = [response.label_annotations[i].description for i in range(len(response.label_annotations))]
    return desc

def getOutput(string):
    with open("input.png", "wb") as fh:
        fh.write(base64.decodebytes(string))
    output = getTextFromImage("input.png")
    os.remove("input.png")
    return output
