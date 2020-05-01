import csv
import editdistance
from spellchecker import SpellChecker


LOAD_FILE = "save/lines/validations_epoch_"
SAVE_FILE = "save/lines/corrected_validations_epoch_"

def correct_words(epoch):
    predictions = []
    targets = []
    spell_checker = SpellChecker()

    with open(LOAD_FILE + str(epoch) + ".csv", "r", newline="") as csvfile:
        # choose other dilimiter, as "," is present in characters
        reader = csv.reader(csvfile, delimiter= "~")
        for row in reader:
            predictions.append(row[1])
            targets.append(row[0])


    altered_predictions = []

    for prediction_line in predictions:
        prediction_words = prediction_line.split()
        corrected_words = []
        for word in prediction_words:
            corrected_words.append(spell_checker.correction(word))
        altered_predictions.append(" ".join(corrected_words))


    altered_num_char_error = 0
    num_char_error = 0
    total_chars = 0


    for i in range(len(targets)):
        altered_prediction = altered_predictions[i]
        prediction = predictions[i]
        target = targets[i]

        altered_distance = editdistance.eval(altered_prediction, target)
        altered_num_char_error += altered_distance

        distance = editdistance.eval(prediction, target)
        num_char_error += distance

        total_chars += len(target)

    CER = 100*num_char_error/total_chars
    altered_CER = 100*altered_num_char_error/total_chars
    #WER =
    print("CER:", CER, "%")
    print("altered CER:", altered_CER, "%")

    for i in range(len(targets)):
        with open(SAVE_FILE + str(epoch) + ".csv", "a", newline="") as csvfile:
            # choose other dilimiter, as "," is present in characters
            writer = csv.writer(csvfile, delimiter= "~")
            writer.writerow([targets[i], altered_predictions[i]])
    with open("save/lines/corrected_CER.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter = ",")
        writer.writerow([str(altered_CER)])

for i in range(51, 71, 1):
    correct_words(i)