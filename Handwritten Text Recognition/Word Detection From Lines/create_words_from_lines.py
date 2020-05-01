import xml.etree.ElementTree as ET
from ElementTree_pretty import prettify
import numpy as np
from scipy import misc
from PIL import Image
import glob
import imageio
import os

"""create words from xml path and img path"""

xmls = os.listdir(
    "C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/data_word_detection/xml_lines_train/")
xmls.sort()

imgs = os.listdir(
    "C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/data_word_detection/lines_train/")
imgs.sort()

height = 0
width = 0

print(imgs[0])
line_count = 1
for i in range(len(imgs)):
    word_count = 1
    #print(imgs[i], xmls[i])
    img = imageio.imread("C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/data_word_detection/lines_train/" + imgs[i])
    print(img.shape)
    height += img.shape[0]
    width += img.shape[1]
    tree = ET.parse("C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/data_word_detection/xml_lines_train/" + xmls[i])
    root = tree.getroot()

    for word in root:
        #print(word.attrib["text"])
        #print(word.attrib["x"])
        #print(word.attrib["width"])
        x1 = int(word.attrib["x"])
        x2 = x1 + int(word.attrib["width"])

        imageio.imwrite(
            "C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/words_test/line" + str(line_count) + "word" + str(word_count) + ".jpg", img[:, int(x1):int(x2)])
        word_count +=1
    
    line_count += 1
    break

print(height / 11507)
print(width / 11507)