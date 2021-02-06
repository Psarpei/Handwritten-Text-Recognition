# Handwritten-Text-Recognition
As part of the project we examine several approaches for handwriting text recognition based on convolutional neural networky and long short-term memories.

<p align="center">                                                                                                                    
    <img align="top" width="400" height="" src="https://upload.wikimedia.org/wikipedia/commons/8/88/Handwritten_text_recognition.jpg">
</p>

All aproaches follow the method to break the image down into the smaller parts:
* lines
* words
* characters

The two best approaches are explained in the written elaboration (only available in german), that you can find between the source code folders of this repository.
On top of that there is a explanation of the object detection approach [YOLOv1](https://arxiv.org/pdf/1506.02640.pdf), and the [End-to-End Trainable Neural Network for Image-based-Sequence Recognition](https://arxiv.org/pdf/1507.05717.pdf) which are used in all approaches.

## General information
<img align="right" width="300" height="" src="https://upload.wikimedia.org/wikipedia/commons/1/1e/Logo-Goethe-University-Frankfurt-am-Main.svg">

**Instructors**
* [Prof. Dr. Visvanathan Ramesh ](http://www.ccc.cs.uni-frankfurt.de/people/), email: mehler@em.uni-frankfurt.de
* [Martin Mundt](https://martin-mundt.com/), email: mmundt@em.uni-frankfurt.de

**Institutions**
* **[Goethe University](http://www.informatik.uni-frankfurt.de/index.php/en/)**
* **[AISEL: AI Systems Engineering Lab](http://www.ccc.cs.uni-frankfurt.de/aisel-ai-systems-engineering-lab/)**

**Project team**
* Martin Ludwig
* [Pascal Fischer](https://github.com/Psarpei)


**Tools**
* Python 3
* PyTorch
* Pillow
* OpenCV

## Project

**Dataset**

<img align="left" width="300" height="" src="https://fki.tic.heia-fr.ch/static/img/a01-122.jpg">

We only use the data of the [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) for training and testing.

The database consists of:

* 657 writers contributed samples of their handwriting
* 1'539 pages of scanned text
* 5'685 isolated and labeled sentences
* 13'353 isolated and labeled text lines
* 115'320 isolated and labeled words

All form, line and word images are provided as PNG files and the corresponding form label files, including segmentation information and variety of estimated parameters, are included in the image files as meta-information in XML format which is described in XML file and XML file format (DTD).

<br/><br/>

**Results**

We compare our best approach with the state-of-the-art [CRNN](https://arxiv.org/pdf/1507.05717.pdf) approach by the character error rate (cer).

| Approach | CER % |
| ---------|:-----:| 
| CRNN     | 5.7   |
| Our best | 10.64 |

**Source Code**

The source code of all approaches are available in the `.pynb` Python formats in the way of google-colab

<a href="https://colab.research.google.com/github/Psarpei/Handwritten-Text-Recognition/blob/master/Handwritten%20Text%20Recognition/Line%20Detection%20From%20Text/LINE_DETECTION.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

