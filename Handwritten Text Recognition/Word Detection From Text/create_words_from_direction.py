import xml.etree.ElementTree as ET
import numpy as np
from scipy import misc
from PIL import Image
import glob
import imageio
import os

"create words from images which are only 25 percent wide and long as the originals"

xmls = os.listdir("C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/data_small/xmls/xml_train/")
imgs = os.listdir(
    "C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/data_small/forms/forms_train_small/")

img_count = 1
percent = 25
for i in range(len(imgs)):
    line_count = 1
    print(imgs[i], xmls[i])
    img = imageio.imread(
        "C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/data_small/forms/forms_train_small/" +
        imgs[i])

    print(img.shape)
    tree = ET.parse(
        "C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/data_small/xmls/xml_train/" +
        xmls[i])
    root = tree.getroot()
    for line in root[1]:  # root[1] catches handwritten part
        print("line#######################################")
        print(line.attrib["text"])
        word_count = 1
        for word in line:
            if (word.tag == "word"):  # there isn't only words lines
                print(word.attrib["text"])
                x1 = 999999999999999
                y1 = 999999999999999
                y2 = 0
                width = 0
                for char in word:
                    x1 = min(x1, int(char.attrib["x"]) * percent / 100)
                    y1 = min(y1, int(char.attrib["y"]) * percent / 100)
                    y2 = max(y2, (int(char.attrib["y"]) + int(char.attrib["height"])) * percent / 100)
                    width = int(char.attrib["x"]) * percent / 100 - x1 + int(char.attrib["width"]) * percent / 100

                x2 = x1 + width
                if(x1 == 999999999999999):
                    continue
                print(int(x1))
                print(int(x2))
                print(int(y1))
                print(int(y2))
                print(img.shape)
                if(int(x1) == int(x2)):
                    x2 += 1
                if(int(y1) == int(y2)):
                    y2 += 1
                imageio.imwrite("C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/words_small_selfmade/img" + str(
                img_count) + "line" + str(line_count) + "word" + str(word_count) + '.jpg', img[int(y1):int(y2), int(x1):int(x2)])
                word_count += 1
        line_count += 1
        # imageio.imwrite("C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analyis_Machine_Intelligence/Abschlussprojekt/lines_cut_selfmade/img" + str(img_count) + "line" + str(line_count) + '.jpg',img[int(b_y): int(b_y + b_h), int(b_x):int(b_x+b_w)])

    img_count += 1