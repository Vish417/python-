# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:06:09 2020

@author: Hamsica
"""

import re
import nltk
import datetime
import pandas as pd

import numpy as np

import unicodedata
import os
import io
from google.cloud import vision
from google.cloud.vision import types
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'apikey.json'
client = vision.ImageAnnotatorClient()


FOLDER_PATH = r'C:\Users\Hamsica\Desktop\google vision'
IMAGE_FILE = '#report1.jpg'
FILE_PATH = os.path.join(FOLDER_PATH, IMAGE_FILE)

with io.open(FILE_PATH, 'rb') as image_file:
    content = image_file.read()

image = vision.types.Image(content=content)
response = client.document_text_detection(image=image)

docText = response.full_text_annotation.text
print(docText)


pages = response.full_text_annotation.pages
for page in pages:
    for block in page.blocks:
        print('block confidence:', block.confidence)

        for paragraph in block.paragraphs:
            print('paragraph confidence:', paragraph.confidence)

            for word in paragraph.words:
                word_text = ''.join([symbol.text for symbol in word.symbols])

                print('Word text: {0} (confidence: {1}'.format(word_text, word.confidence))

                for symbol in word.symbols:
                    print('\tSymbol: {0} (confidence: {1}'.format(symbol.text, symbol.confidence))
