import xml.etree.ElementTree as ET
from ElementTree_pretty import prettify
import numpy as np
from scipy import misc
from PIL import Image
import glob
import imageio
import os

"""create xmls for lines from original xmls and imgs"""

xmls = os.listdir(
    "C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/xml_train")
imgs = os.listdir(
    "C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/forms_train/")

comment = ET.Comment('Generated from Pascal Fischer, Goethe University, Frankfurt am Main, Praktikum Pattern Analysis and Machine Intelligence')

img_count = 1
for i in range(len(imgs)):
    line_count = 1
    print(imgs[i], xmls[i])
    img = imageio.imread("C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/forms_train/" + imgs[i])
    print(img.shape)
    tree = ET.parse("C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/xml_train/" + xmls[i])
    root = tree.getroot()
    for line in root[1]:  # root[1] catches handwritten part
        word_count = 1
        x_line = 999999999999999
        text_line = line.attrib["text"]
        top = ET.Element("line")
        top.set("text", text_line)
        top.append(comment)
        #print(line_text)

        for word in line:
            x = 999999999999999
            if (word.tag == "word"):  # there isn't only words lines
                for char in word:
                    x_line = min(x_line, int(char.attrib["x"]))

        for word in line:
            x = 999999999999999
            if (word.tag == "word"):  # there isn't only words lines
                text_word = word.attrib["text"]
                for char in word:
                    x = min(x, int(char.attrib["x"]))
                    width = int(char.attrib["x"]) - x + int(char.attrib["width"])

                if (x == 999999999999999):
                    continue

                x -= x_line

                child = ET.SubElement(top, 'word')
                child.set("text", text_word)
                child.set("x", str(x))
                child.set("width", str(width))

        f = open("C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/xml_lines_train/img" + str(img_count) + "line" + str(line_count) + ".xml", "a")
        f.write(prettify(top))
        f.close()
        line_count += 1
    img_count += 1