import torch
import torchvision
from torchvision import transforms
from PIL import Image
from os import listdir
import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
from torchnet import meter

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 5)  # input features, output features, kernel size
        self.mp1 = nn.MaxPool2d(2, 2)  # kernel size, stride
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.mp2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 13 * 13, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv + ReLU + max pooling for two layers
        x = self.mp1(F.relu(self.conv1(x)))
        x = self.mp2(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_data(path, transform, chars, batch_size):
    "creates data from images to tensor"

    data_list = []
    target_list = []
    data = []
    files = listdir(path)

    for i in range(len(listdir(path))):
        f = random.choice(files)
        files.remove(f)
        if(f[5] not in chars): #we dont want all charackter types at the moment
            continue
        img = Image.open(path + f)
        img_tensor = transform(img)  # (3,256,256)
        data_list.append(img_tensor)
        #target = torch.LongTensor([int(f[5] == char) for char in chars]) one hot vector
        target = torch.LongTensor([i for i in range(len(chars)) if f[5] == chars[i]])
        #print(f, target)
        target_list.append(target)
        if len(data_list) >= batch_size:
            data.append((torch.stack(data_list), torch.stack(target_list)))
            data_list = []
            target_list = []
            print('Loaded batch ', len(data), 'of ', int(len(listdir(path)) / batch_size))
            print('Percentage Done: ', 100 * len(data) / int(len(listdir(path)) / batch_size), '%')
    return data

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """
    Evaluates a model's top k accuracy

    Parameters:
        output (torch.autograd.Variable): model output
        target (torch.autograd.Variable): ground-truths/labels
        topk (list): list of integers specifying top-k precisions
            to be computed

    Returns:
        float: percentage of correct predictions
    """

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(train_data, model, criterion, optimizer, device):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        train_data (torch tensor): The trainset
        model (torch.nn.module): Model to be trained
        criterion (torch.nn.criterion): Loss function
        optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
        device (string): cuda or cpu
    """

    # create instances of the average meter to track losses and accuracies
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    # iterate through the dataset loader
    i = 0
    for (inp, target) in train_data:
        # transfer inputs and targets to the GPU (if it is available)
        inp = inp.to(device)
        target = target.to(device)

        # compute output, i.e. the model forward
        output = model(inp)

        #print("target", target.size(), "output", output.size())
        # calculate the loss

        loss = criterion(output, target.squeeze())

        # measure accuracy and record loss and accuracy
        prec1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inp.size(0))
        top1.update(prec1.item(), inp.size(0))

        # compute gradient and do the SGD step
        # we reset the optimizer with zero_grad to "flush" former gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print the loss every 100 mini-batches
        if i % 100 == 0:
            print('Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                loss=losses, top1=top1))
        i += 1

def validate(val_data, model, criterion, device):
    """
    Evaluates/validates the model

    Parameters:
        val_data (torch.Tensor): The validation or testset
        model (torch.nn.module): Model to be evaluated/validated
        criterion (torch.nn.criterion): Loss function
        device (string): cuda or cpu
    """

    # create instances of the average meter to track losses and accuracies
    losses = AverageMeter()
    top1 = AverageMeter()

    #confusion = meter.ConfusionMeter(len(val_loader.dataset.class_to_idx))

    # switch to evaluate mode
    # (this would be important for e.g. dropout where stochasticity shouldn't be applied during testing)
    model.eval()

    # avoid computation of gradients and necessary storing of intermediate layer activations
    with torch.no_grad():
        # iterate through the dataset loader
        i = 0
        for (inp, target) in val_data:
            # transfer to device
            inp = inp.to(device)
            target = target.to(device)

            # compute output
            output = model(inp)

            # compute loss
            loss = criterion(output, target.squeeze())

            # measure accuracy and record loss and accuracy
            prec1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inp.size(0))
            top1.update(prec1.item(), inp.size(0))

            # add to confusion matrix
            #confusion.add(output.data, target)

    print(' * Validation accuracy: Prec@1 {top1.avg:.3f} '.format(top1=top1))

def main():
    path_train = "C:/Users/Pasca/Dropbox/ML_Praktikum/chars/"

    weight_path = "C:/Users/Pasca/Dropbox/ML_Praktikum/CNN_weights/"

    batch_size = 64

    """
    normalize = transforms.Normalize(     #can we use later
       mean=[],
       std=[]
    )
    """

    transform = transforms.Compose([transforms.Resize((64, 64)),
                                     transforms.ToTensor()])

    chars = "abcdefghijklmnopqrstuvwxyz0123456789,.()"

    train_data = create_data(path_train, transform, chars, batch_size)

    print(train_data[0][0].size(), len(train_data[0][1]))
    print(len(chars))

    # set a boolean flag that indicates whether a cuda capable GPU is available
    is_gpu = torch.cuda.is_available()
    print("GPU is available:", is_gpu)
    print("If you are receiving False, try setting your runtime to GPU")

    # set the device to cuda if a GPU is available
    device = torch.device("cuda" if is_gpu else "cpu")

    # new model instance
    model = CNN(len(chars)).to(device)
    #model = torch.load("Hier den Path einfügen")

    # loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # optimize
    total_epochs = 10
    for epoch in range(total_epochs):
        print("EPOCH:", epoch + 1)
        print("TRAIN")
        train(train_data, model, criterion, optimizer, device)
        #print("VALIDATION")
        #validate(val_dataset, model, criterion, device)
        torch.save(model, weight_path + "weights_" + str(epoch) + ".pt")

if __name__ == "__main__":
    main()