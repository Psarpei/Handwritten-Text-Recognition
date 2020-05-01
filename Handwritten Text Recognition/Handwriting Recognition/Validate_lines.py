import editdistance
import os
import random
import numpy as np
from CNN_BILSTIM_MODEL_LINES import CNN_BiLSTM
import torch
from torchvision import transforms
from PIL import Image
from Dataloader_Lines import DataLoader
from spellchecker import SpellChecker

loader = DataLoader(file_path='data/', batch_size=1, img_size=(64,512), max_text_len=256)
file_path = "prediction_lines/prediction_lines/"

device = "cpu"
model = CNN_BiLSTM(loader.char_list, 1, device).to(device)
model.load_state_dict(torch.load("save/lines/CNN_BiLSTM_after_epoch_62.pt"))

model.eval()
forms = os.listdir(file_path)

spell_checker = SpellChecker()

num_char_err = 0
altered_char_err = 0
num_char_total = 0


for form in forms:


    text_file = open(file_path + form + "/text/text.txt")

    targets = text_file.readlines()

    target_texts = []
    for target in targets:
        target_texts.append(target[:-1])
    target_text = "".join(target_texts)

    num_char_total += len(target_text)
    lines = os.listdir(file_path+form + "/lines/")
    lines_mod = []
    for line in lines:
        if line[5].isdigit():
            lines_mod.append(line[:4] + "9" + line[5:])
        else:
            lines_mod.append(line)
    lines_mod = np.sort(lines_mod)
    preds = []
    for line in lines_mod:
        if len(line) == 10:
            line = line[:4] + "1" + line[5:]
        img = Image.open(file_path + form + "/lines/" +line)

        img = transforms.Resize((64, 512))(img)

        img = transforms.ToTensor()(img)

        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]).to(device)

        pred = model(img).cpu()
        pred = pred.permute(1,0,2).cpu().data.numpy()
        pred = model.prediction_to_string(pred)
        preds.append(pred)
    preds = "".join(preds)

    altered_predictions = []

    for prediction_line in lines_mod:
        prediction_words = prediction_line.split()
        corrected_words = []
        for word in prediction_words:
            corrected_words.append(spell_checker.correction(word))
        altered_predictions.append(" ".join(corrected_words))

    altered_predictions = "".join(altered_predictions)

    dist = editdistance.eval(preds, target_text)
    altered_dist = editdistance.eval(altered_predictions, target_text)
    #print(preds)
    #print(target_text)
    print(dist)
    num_char_err += dist
    altered_char_err += altered_dist

char_error_rate = num_char_err / num_char_total
altered_CER = altered_char_err / num_char_total
print("CER:", char_error_rate)
print("altered CER:", altered_CER)

