import xml.etree.ElementTree as ET
import numpy as np
from scipy import misc
from PIL import Image
import glob
import imageio
import os

xmls = os.listdir("C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/xml_test")
imgs = os.listdir("C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/forms_test/")

img_count = 1
percent = 25
for i in range(len(imgs)):
    line_count = 1
    print(imgs[i], xmls[i])
    img = imageio.imread("C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/forms_test/" + imgs[i])

    print(img.shape)
    tree = ET.parse("C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/xml_test/" + xmls[i])
    root = tree.getroot()
    for line in root[1]: #root[1] catches handwritten part
        x1 = 999999999999999
        y1 = 999999999999999
        y2 = 0
        width = 0
        print(line.attrib["text"])
        for word in line:
            if(word.tag == "word"): #there isn't only words lines
                for char in word:
                    x1 = min(x1, int(char.attrib["x"]))
                    y1 = min(y1, int(char.attrib["y"]))
                    y2 = max(y2, int(char.attrib["y"]) + int(char.attrib["height"]))
                    width = int(char.attrib["x"]) - x1 + int(char.attrib["width"])

        x2 = x1 + width

        imageio.imwrite("C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analysis_Machine_Intelligence/Abschlussprojekt/lines_test/img" + str(img_count) + "line" + str(line_count) + '.jpg',img[int(y1):int(y2),int(x1):int(x2)])
        
        line_count += 1
    img_count +=1