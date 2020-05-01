import xml.etree.ElementTree as ET
import numpy as np
from scipy import misc
from PIL import Image
import glob
import imageio
import os

"""create chars from imgs and xmls
   !!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!! 
   this doesn't work really good,
   because the data annotation is very faulty.
   The output generates so much errors and
   doesn't generate only words where the annotation
   length is like the word length!
   !!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!
"""

imgs = os.listdir("C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analyis_Machine_Intelligence/Abschlussprojekt/forms")
xmls = os.listdir("C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analyis_Machine_Intelligence/Abschlussprojekt/xml")

chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.-;:_!/?#+*ß=&%$ '()[]" + '"'

char_counter = {}
for char in chars:
    char_counter[char]=0
print(char_counter)

for i in range(len(imgs)):
    print(imgs[i], xmls[i])
    img = imageio.imread("C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analyis_Machine_Intelligence/Abschlussprojekt/forms/" + imgs[i])

    tree = ET.parse("C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analyis_Machine_Intelligence/Abschlussprojekt/xml/" + xmls[i])
    root = tree.getroot()

    for line in root[1]: #root[1] catches handwritten part
        for word in line:
            if(word.tag == "word"): #there isn't only words lines
                if(len(word) == len(word.attrib["text"])):
                    print(word.tag, word.attrib["text"])
                    count = 0
                    for char in word:
                        c = word.attrib["text"][count]
                        if(c in [" ", "/", '"', "'", "*","?", "!"]):
                            continue
                        num = char_counter[c]
                        name = "char" + "_" + c + "_" + str(num)
                        x1 = int(char.attrib["x"])
                        y1 = int(char.attrib["y"])
                        x2 = x1 + int(char.attrib["width"])
                        y2 = y1 + int(char.attrib["height"])
                        imageio.imwrite('chars/' + name + '.jpg',img[y1:y2,x1:x2])
                        #print(char.tag, char.attrib["height"])
                        count += 1
                        char_counter[c] += 1
