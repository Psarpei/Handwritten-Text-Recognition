import editdistance
from CNN_BILSTIM_MODEL_LINES import CNN_BiLSTM
from Dataloader_Lines import DataLoader
#from Dataloader import DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import csv
import numpy as np
import pickle

BATCH_SIZE = 8
IMAGE_SIZE = (64, 512)
MAX_TEXT_LEN = 128
LOAD = False
LOAD_FILE = "CNN_BiLSTM_after_pretraining.pt"
LOAD_FOLDER = "save/lines/"
SAVE_FOLDER = "save/lines/"

DIST_SAMPLE = "data/words/a01/a01-063u/a01-063u-09-02.png"

def string_to_tensor(string, char_list):
    """A helper function to create a target-tensor from a target-string

    params:
        string - the target-string
        char_list - the ordered list of characters

    output: a torch.tensor of shape (len(string)). The entries are the 1-shifted
        indices of the characters in the char_list (+1, as 0 represents the blank-symbol)
    """
    target = []
    for char in string:
        pos = char_list.index(char) + 1
        target.append(pos)

    result = torch.tensor(target, dtype=torch.int32)
    return result


def pretrain(model, optimizer, batches):
    """Pretrains the model on a certain set of images.

    params:
        """
    #model.train()
    device = model.device

    loss_function = nn.CTCLoss(zero_infinity=True)

    num_batches = len(batches)

    epochs_since_improvement = 0
    #best_num_5_or_less = 0

    # Pretraining is stopped, when mean loss improved by less than 0.1%
    # (or doesn't improve at all)
    epsilon = 0.001
    mean_losses = []
    CERs = []
    num_epochs = 0
    while True:
        # save losses and editdistances in lists
        losses = []
        dists = []
        all_target_lengths = []
        model.train()
        for batch in batches:
            # zero cached gradients
            optimizer.zero_grad()
            # get images from the batch
            imgs = batch.imgs.to(device)
            # get prediction from the model
            pred_train = model(imgs)
            # get target-texts and lenghts. Concatenate the target-texts
            target_texts = batch.target_texts
            target_lengths = torch.tensor([len(text) for text in target_texts]).to(device)
            target_texts = "".join(target_texts)
            # get tensors from target-string
            targets = string_to_tensor(target_texts, model.chars).to(device)
            # input lengths is always the same
            input_lengths = torch.tensor([pred_train.size(0)] * BATCH_SIZE).int().to(device)

            # give everything to the loss-function, save loss
            loss = loss_function(pred_train, targets, input_lengths, target_lengths)
            losses.append(loss.item())
            loss.backward()

            # clip gradients, as they tend to explode in RNN's
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        # evaluate by
        model.eval()
        for batch in batches:
            # get a new prediction. In evaluation mode, there is no dropout
            imgs = batch.imgs.to(device)
            target_texts = batch.target_texts
            pred_eval = model(imgs)
            target_lengths = []

            pred_strings = model.decode(pred_eval)
            for i in range(len(pred_strings)):
                dist = editdistance.eval(pred_strings[i], target_texts[i])
                all_target_lengths.append(len(target_texts[i]))
                dists.append(dist)


        mean_loss = np.mean(losses)
        mean_losses.append(mean_loss)
        print("Loss:", mean_loss)
        CER = 100*np.sum(dists) / np.sum(all_target_lengths)
        CERs.append(CER)
        print("CER:", CER, "%")

        num_epochs += 1
        # save losses and CER's on train-set in CSV
        with open(SAVE_FOLDER + "pretrain_losses_and_CER.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter = ",")
            writer.writerow([str(mean_loss), str(CER)])

        # only start comparing after 2nd epoch
        if num_epochs >= 2:
            diff = mean_losses[-2] - mean_losses[-1]
            print(diff / mean_losses[-2] )
            if diff / mean_losses[-2] < epsilon:
                return  mean_losses, CERs



def train(model, loader):
    "train NN"

    # start with a validation, which is after pretraining


    char_list = loader.char_list
    epoch = 0
    best_character_error_rate = float('inf') # best valdiation character error rate
    no_improvement_since = 0 # number of epochs no improvement of character error rate occured
    early_stop = 20 # stop training after this number of epochs without improvement
    #
    char_error_rate = validate(model, loader, epoch)


    device = model.device
    optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.001, weight_decay = 0.0001)
    loss_function = nn.CTCLoss(zero_infinity=True)
    mean_losses = []
    min_dist = float('inf')
    #validate(model, loader)
    while True:
        epoch += 1
        losses = []
        dists = []
        print('Epoch:', epoch)

        # train
        print('Start Training')
        loader.train_set()
        model.train()
        perc = 5
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            if (100*iter_info[0]/iter_info[1]) > perc:
                print(str(perc) + "% of epoch done" )
                perc += 5
            #print('Batch:', iter_info[0],'/', iter_info[1])
            #torch.cuda.empty_cache()
            batch = loader.get_next()
            optimizer.zero_grad()
            imgs = batch.imgs.to(device)
            target_texts = batch.target_texts
            target_lengths = torch.tensor([len(text) for text in target_texts]).to(device)

            target_texts = "".join(target_texts)
            targets = string_to_tensor(target_texts, char_list).to(device)

            predictions = model(imgs)
            # shape is (time_steps, batch_size, num_chars)
            #pred_string = model.prediction_to_string(predictions.permute(1,0,2).cpu().data.numpy())
            #dist = editdistance.eval(pred_string, target_texts)
            #dists.append(dist)

            prediction_length = torch.tensor([predictions.size(0)] * BATCH_SIZE).int().to(device)

            loss = loss_function(predictions, targets, prediction_length, target_lengths)
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            loss_value = loss.item()
            losses.append(loss_value)



            optimizer.step()
        print("Epoch", epoch, "end")
        #if (sum(dists) < min_dist):
         #   min_dist = sum(dists)
          #  epochs_since_improvement = 0
        #else:
         #   epochs_since_improvement += 1
        #print("Total edit distance:", sum(dists), "Min edit distance:",
         #     str(min_dist))

        # validate
        char_error_rate = validate(model, loader, epoch)

        mean_loss = np.mean(losses)
        mean_losses.append(mean_loss)
        # save the model
        torch.save(model.state_dict(), SAVE_FOLDER + "CNN_BiLSTM_after_epoch_" + str(epoch) + ".pt")

        # if best validation accuracy so far, save model parameters
        if char_error_rate < best_character_error_rate:
            print('Character error rate improved')
            best_character_error_rate = char_error_rate
            no_improvement_since = 0

            #open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))

        else:
            print('Character error rate not improved')
            no_improvement_since += 1

		# stop training if no more improvement in the last x epochs
        if no_improvement_since >= early_stop:
            print('No more improvement since %d epochs. Training stopped.' % early_stop)
            break


        save_loss_and_error_rate(mean_loss, char_error_rate)
        save_losses(losses)


def save_losses(losses):
    with open(SAVE_FOLDER + "losses.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter = ",")
        writer.writerow(losses)
def save_loss_and_error_rate(mean, error_rate):
    with open(SAVE_FOLDER +"mean_loss_and_error_rate.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter = ",")
        writer.writerow([str(mean), str(error_rate*100)])

def visualize_prediction_course(model):

    img = Image.open(DIST_SAMPLE)
    img = transforms.Resize(IMAGE_SIZE)(img)
    img = transforms.ToTensor()(img)


    img = img.reshape(1, 1, 64, 512)
    log_probs = model(img.to(model.device))
    log_probs = log_probs.permute(1,0,2).cpu().data.numpy()
    prediction_raw = model.prediction_to_raw_string(log_probs)
    pred_string = model.prediction_to_string(log_probs)
    print(prediction_raw)
    print(pred_string)
    print("")
    with open(SAVE_FOLDER + "prediction_course.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter= ",")
        writer.writerow([prediction_raw,pred_string])

def visualize_blank_distribution(model):


    img = Image.open(DIST_SAMPLE)
    img = transforms.Resize(IMAGE_SIZE)(img)
            #img.show()
    img = transforms.ToTensor()(img)


    img = img.reshape(1, 1, IMAGE_SIZE[0], IMAGE_SIZE[1])

    log_probs = model(img.to(model.device))
    # log_probs of shape (seq_len, batch_size, num_chars)
    for char in range(model.num_chars+1):
        dist = []
        for i in range(log_probs.shape[0]):
            dist.append(log_probs[i][0][char].item())

        dist = np.exp(dist)
        with open(SAVE_FOLDER + str(char) + "_distribution.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter= ",")
            writer.writerow(dist)


def validate(model, loader, epoch, correction=False):
    "validate NN"
    print('Validate NN')
    model.eval()
    loader.validation_set()
    num_char_err = 0
    num_char_total = 0
    num_lines_OK = 0
    num_lines_total = 0

    #visualize_blank_distribution(model)
    #visualize_prediction_course(model)
    #i = 0
    #perc = 0
    preds = []
    targets = []
    while loader.has_next():
        #i+= BATCH_SIZE
        #if i*100//len(loader.validation_samples) > perc:
         #   print(perc, "% of validation done")
          #  perc += 10
        iter_info = loader.get_iterator_info()
        print('Batch:', iter_info[0],'/', iter_info[1])
        batch = loader.get_next()

        prediction = model(batch.imgs.to(model.device)).cpu()

        #pred = prediction.permute(1,0,2).cpu().data.numpy()
        pred_strings = model.decode(prediction)

        target_texts = batch.target_texts
        for i in range(len(pred_strings)):
        #print('Ground truth -> Recognized')
            num_lines_OK += 1 if target_texts[i] == pred_strings[i] else 0
            num_lines_total += 1
            dist = editdistance.eval(pred_strings[i], target_texts[i])
            num_char_err += dist
            num_char_total += len(target_texts[i])
            preds.append(pred_strings[i])
            targets.append(target_texts[i])
            with open(SAVE_FOLDER + "validations_epoch_" + str(epoch) + ".csv", "a", newline="") as csvfile:
                # choose other dilimiter, as "," is present in characters
                writer = csv.writer(csvfile, delimiter= "~")
                writer.writerow([target_texts[i], pred_strings[i]])

        #if pred != "":
        #    print('[OK]' if dist==0 else '[ERR:%d]' % dist, target , '->', pred)




	# print validation result
    char_error_rate = num_char_err / num_char_total
    line_accuracy = num_lines_OK / num_lines_total
    print('Character error rate: %f%%. Line accuracy: %f%%.' % (char_error_rate*100.0, line_accuracy*100.0))
    return char_error_rate


#loader = DataLoader(file_path='data/', batch_size=BATCH_SIZE, img_size=IMAGE_SIZE, max_text_len=MAX_TEXT_LEN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
#model = CNN_BiLSTM(loader.char_list, BATCH_SIZE, device).to(device)





if LOAD:
    loader = pickle.load(open(LOAD_FOLDER + "loader.p", "rb"))
    model = CNN_BiLSTM(loader.char_list, BATCH_SIZE, device).to(device)
    model.load_state_dict(torch.load(LOAD_FOLDER + LOAD_FILE))
    #imgs = pickle.load(open(SAVE_FOLDER + "imgs.p", "rb"))
    #target_texts = pickle.load(open(SAVE_FOLDER + "target_texts.p", "rb"))
    #targets = pickle.load(open(SAVE_FOLDER + "targets.p", "rb"))
    #target_lengths = pickle.load(open(SAVE_FOLDER + "target_lengths.p", "rb"))
    #input_lengths = pickle.load(open(SAVE_FOLDER + "input_lengths.p", "rb"))

#optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.001, weight_decay = 0.0001)


# uncomment to begin with pretraining

# loader = DataLoader(file_path='data/', batch_size=BATCH_SIZE, img_size=IMAGE_SIZE, max_text_len=MAX_TEXT_LEN)
# model = CNN_BiLSTM(loader.char_list, BATCH_SIZE, device).to(device)
# optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.001, weight_decay = 0.0001)
# batches = []
# for i in range(64):
#     batches.append(loader.get_next())

# mean_losses, CERs = pretrain(model, optimizer, batches)
# torch.save(model.state_dict(), SAVE_FOLDER + "CNN_BiLSTM_after_pretraining.pt")
# pickle.dump(loader, open(SAVE_FOLDER + "loader.p", "wb"))


# uncomment to begin after pretraining
# loader = pickle.load(open(LOAD_FOLDER + "loader.p", "rb"))
# model = CNN_BiLSTM(loader.char_list, BATCH_SIZE, device).to(device)
# model.load_state_dict(torch.load(LOAD_FOLDER + LOAD_FILE))
#optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.001, weight_decay = 0.0001)
# train(model, loader)



