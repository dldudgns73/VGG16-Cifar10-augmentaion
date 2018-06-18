import argparse
import torch
import pickle as pkl
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from vgg import VGG
from torch.autograd import Variable

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default = 'train_model/base_model')
#parser.add_argument('--model_dir', default = 'train_model/RandomHorizontalFlip_RandomCrop32_model')
#parser.add_argument('--model_dir', default = 'train_model/RandomHorizontalFlip_RandomCrop32_Normalize_model')

parser.add_argument('--use_cuda', default=False)
parser.add_argument("--eval", default=True)

parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--lrate', type = float, default = 0.005)
parser.add_argument('--dropout', type = float, default = 0.3)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--decay_period', type = int, default = 20)
parser.add_argument('--decay', type = float, default=0.5)

args = parser.parse_args()

def train() :
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size = 32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = VGG(vars(args))
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lrate,
                                momentum = 0.9,
                                weight_decay=5e-4)

    if args.use_cuda :
        model = model.cuda()


    if args.eval :
        model.load_state_dict(torch.load(args.model_dir))
        model.eval()
        accuracy = model.evaluate(testloader)
        exit()

    total_size = len(trainloader)
    lrate = args.lrate
    best_score = 0.0
    scores = []
    for epoch in range(1, args.epochs+1) :
        model.train()
        for i, (image, label) in enumerate(trainloader) :

            loss = model(image, label)
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0 :
                print('Epoch = %d, step = %d / %d, loss = %.5f lrate = %.5f' %(epoch, i, total_size, loss, lrate))

        model.eval()
        accuracy = model.evaluate(testloader)
        scores.append(accuracy)

        with open(args.model_dir + "_scores.pkl", "wb") as f :
            pkl.dump(scores, f)

        if best_score < accuracy :
            best_score = accuracy
            print('saving %s ...' % args.model_dir)
            torch.save(model.state_dict(), args.model_dir)

        if epoch % args.decay_period == 0 :
            lrate *= args.decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lrate

if __name__ == '__main__' :
    train()