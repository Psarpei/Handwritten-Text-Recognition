
import os
import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


class Sample:
    """sample from the dataset
    params:
        target_text: Str - the text that is shown by the image
        file_path: Str - the relative filePath of the image
    """
    def __init__(self, target_text, file_path):
        self.target_text = target_text
        self.file_path = file_path


class Batch:
    "batch containing images and ground truth texts"
    def __init__(self, target_texts, imgs):
        self.imgs = torch.stack(imgs, axis=0)
        self.target_texts = target_texts


class DataLoader:
    "loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database"

    def __init__(self, file_path="data/", batch_size=1, img_size=[128, 2048], max_text_len=256):
        """loader for dataset at given location, preprocess images and text according to parameters.
        params:
            file_path: Str       - folder that contains the data
            batch_size: Int      - number of samples per batch
            img_size: (Int, Int) - (height, width) of the produced images
            max_text_len: Int    -

        """

        # filePath needs to be a folder
        assert file_path[-1]=='/'

        self.current_index = 0
        self.batch_size = batch_size
        self.img_size = img_size
        self.samples = []

        # metadata for words in words.txt
        f=open(file_path+'lines.txt')


        chars = set()
        bad_samples = []
        #bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
        for line in f:
            # ignore comment line
            if not line or line[0]=='#':
                continue
            # line = (name, status, graylevel, components, (x y w h), grammatical tag, ground_truth)
            line_split = line.strip().split(' ')
            assert len(line_split) >= 9

            # get path + name of a file
            file_name_split = line_split[0].split('-')
            file_name = file_path + 'lines/' + file_name_split[0] + '/' + file_name_split[0] + '-' + file_name_split[1] + '/' + line_split[0] + '.png'

			# GT text are columns starting at 9, cut off words that are too long
            target_text = line_split[8:]
            target_text = "".join(target_text)

            target_text = target_text.replace("|", " ")
            # get all characters present in the dataset.
            chars = chars.union(set(list(target_text)))

			# check if image is not empty
            if not os.path.getsize(file_name):
                bad_samples.append(line_split[0] + '.png')
                continue

            # put sample into list
            self.samples.append(Sample(target_text, file_name))

        # some images in the IAM dataset are known to be damaged, don't show warning for them
        #if set(bad_samples) != set(bad_samples_reference):
         #   print("Warning, damaged images found:", bad_samples)

         #print("Damaged images expected:", bad_samples_reference)

        # split into training and validation set: 95% - 5%
        split_index = int(0.95 * len(self.samples))
        self.train_samples = self.samples[:split_index]
        self.validation_samples = self.samples[split_index:]

        # put words into lists
        self.train_words = [x.target_text for x in self.train_samples]
        self.validation_words = [x.target_text for x in self.validation_samples]

        # number of randomly chosen samples per epoch of training
        self.train_samples_per_epoch = len(self.train_samples)//2

        # start with train set
        self.train_set()

    	# sorted list of all chars in dataset
        self.char_list = sorted(list(chars))


    def truncate_label(self, text, max_text_len):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i-1]:
                cost += 2
            else:
                cost += 1
            if cost > max_text_len:
                return text[:i]
            return text


    def train_set(self):
        "switch to randomly chosen subset of training set"
        #self.dataAugmentation = True
        self.current_index = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples[:self.train_samples_per_epoch]


    def validation_set(self):
        "switch to validation set"
        #self.dataAugmentation = False
        self.current_index = 0
        self.samples = self.validation_samples


    def get_iterator_info(self):
        "current batch index and overall number of batches"
        return (self.current_index // self.batch_size + 1, len(self.samples) // self.batch_size)


    def has_next(self):
        "iterator"
        return self.current_index + self.batch_size <= len(self.samples)


    def get_next(self):
        "iterator"
        batch_range = range(self.current_index, self.current_index + self.batch_size)
        target_texts = [self.samples[i].target_text for i in batch_range]

        imgs = []
        for i in batch_range:
            img = Image.open(self.samples[i].file_path)
            #img = img.convert("RGB")

            img = transforms.Resize(self.img_size)(img)
            #img.show()
            img = transforms.ToTensor()(img)
            #img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
             #                    std=[0.229, 0.224, 0.225])(img)
            imgs.append(img)
        self.current_index += self.batch_size
        return Batch(target_texts, imgs)


