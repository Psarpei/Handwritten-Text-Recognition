# Handwritten-Text-Recognition
As part of the project we examine several approaches for recognizing text in images and predicting the whole digital text.

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
<img align="right" width="300" height="" src="https://upload.wikimedia.org/wikipedia/commons/1/1e/Logo-Goethe-University-Frankfurt-am-Main.svg">;

**Instructors**
* [Prof. Dr. Visvanathan Ramesh ](http://www.ccc.cs.uni-frankfurt.de/people/), email: mehler@em.uni-frankfurt.de
* [Martin Mundt](https://martin-mundt.com/), email: mmundt@em.uni-frankfurt.de

**Institutions**
* **[Goethe University](http://www.informatik.uni-frankfurt.de/index.php/en/)**
* **[TTLab - Text Technology Lab](https://www.texttechnologylab.org/)**

**Project team**
* Martin Ludwig
* [Pascal Fischer](https://github.com/Psarpei)


## Project
The model is able to recognizing the followig logical document structures
* ```(t``` - text start
* ```(s``` - sentence start
* ```(seg``` - segment start
* ```(w``` - word start
* ```(c``` - char start
* ```)``` end of logical document structure
* ```Ti``` - sentence/segment depth will be measured recursive 

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

**Training results**

<img align="center" width="500" height="" src="https://upload.wikimedia.org/wikipedia/commons/0/0a/Train_example2.png">

**Vaidation results**

<img align="center" width="500" height="" src="https://upload.wikimedia.org/wikipedia/commons/b/bb/Test_example2.png">

**Prediction**

The model is able to predict the output in the usually format like in example_predict.txt. or in the .xml format like in example_XML.tei.

In following ```DATANAME``` is a free selectable name. 

There are to options to make a prediction:
* You have the ground truth of a text and you want to predict the rnng-output as ```.tei``` file as well as the evaluation. For this you have to save your ground_truth data as ```DATANAME_Ground_Truth.txt```.
* You only want to predict the rnng-output. For this you have to save your text in the file ```DATANAME.txt``` as plain text.

In both cases you have to save your files in the directory ```/PLACE_YOUR_FILES_HERE```

To make a prediciton (and get the evaluation results) you only have to navigate to the directory of the ```predict.sh``` file and execute 

    ./predict.sh DATEINAME 

The following files are generated in the ```/PLACE_YOUR_FILES_HERE directory```

* ```DATEINAME_graminput.txt```: The rnng input file 
* ```DATEINAME_predict.txt```: The prediction in grammar format.
* ```DATEINAME.tei```: This is the final rnng out file which contains the prediction in .tei format.
* ```DATEINAME.txt```: The purely input text without grammar format is saved here (if not available).
* ```DATEINAME_Ground_Truth.txt```: This file only exists if, you saved a ground truth file here.
* ```DATEINAME_evaluation.txt```: Here you can find the evaluation results (this file only exists if you have saved a ground truth file).
