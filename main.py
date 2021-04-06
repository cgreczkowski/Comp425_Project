
# import required libraries, DO NOT MODIFY!
import torch
import torchvision
from torchvision.datasets import ImageFolder,KMNIST
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn
import numpy as np

### TODO: set random seed to your Student ID
random_seed = 40001600
torch.manual_seed(random_seed);

# datasets hyper parameters
batch_size = 20
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

# Initialize kmnist train and test datasets
# These two lines will download the datasets in a folder called KMNIST.
# The folder will be written in the same directory as this script.
# The download will occur once. Subsequent executions will not re-download the datasets if they exist.
kmnist_train_set = KMNIST(root='.',
                         train=True,
                         download=False,
                         transform=train_transform)
kmnist_test_set = KMNIST(root='.',
                         train=False,
                         download=False,
                         transform=test_transform)

# Initialize kmnist train and test data loaders.
kmnist_train_loader = torch.utils.data.DataLoader(kmnist_train_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=True)
kmnist_test_loader = torch.utils.data.DataLoader(kmnist_test_set,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=True)
#
# ### TODO: visualize a sample image and corresponding label from KMNIST
# def matplotlib_imshow(img):
#     i, l = img
#     i = i.mean(dim=0)
#     npimg = i.numpy()
#     plt.imshow(npimg, cmap="Greys")
#     print(l)
#
# matplotlib_imshow(kmnist_train_set[0])
#
# def Sigmoid(x):
#     """ Identity activation function
#     Args:
#         x (torch.tensor)
#     Return:
#         torch.tensor: a tensor of shape of x
#     """
#     # For some reason, this formula causes accuracy to suddenly drop around epoch 3, using torch for now..
#     # return 1.0 / (1.0 +torch.exp(-x))
#     return torch.sigmoid_(x)
#
#
# def ReLU(x):
#     """ ReLU activation function
#     Args:
#         x (torch.tensor)
#     Return:
#         torch.tensor: a tensor of shape of x
#     """
#
#     z = torch.zeros(x.size(), device='cuda')
#     return torch.where(x > 0, x, z)
#
# def Identity(x):
#     """ Identity activation function
#     Args:
#         x (torch.tensor)
#     Return:
#         torch.tensor: a tensor of shape of x
#     """
#     return x
#
# def Softmax(x,dim):
#     """ Softmax function
#     Args:torch.log(
#         x (torch.tensor): inputs tensor of size (B,F)
#         dim (int): A dimension along which Softmax will be computed
#     Return:
#         torch.tensor: a tensor of shape of x
#     """
#     mat = torch.nn.Softmax(dim)
#     return mat(x)
#
#
def CE_loss(predictions,labels):
    """ Cross entropy loss
    Args:
        predictions (torch.tensor): tensor of shape of (B,C)
        labels (torch.tensor): tensor of shape of (B,1)
    Returns:
        torch.tensor: a tensor of shape of (1,)
    """

    ### TODO: Fill out this function
    loss = nn.CrossEntropyLoss().cuda()
    output = loss(predictions, labels)
    return output
#
#
# params = {}
#
#
# class my_nn:
#     def __init__(self, layers_dim, layers_activation=None, device='cuda'):
#         """ Initialize network
#         Args:
#             layers_dims (List of ints): list of Size of each layer of the network
#                                         [inputs,layer1,...,outputs]
#             layers_activation (List of strings): list of activation function for each hidden layer
#                                         of the network[layer1,...,outputs]
#             device (str): a device that will be used for computation
#                 Default: 'cpu'
#
#         """
#         self.layers_activation = layers_activation
#         self.params = {}
#         self.num_layers = len(layers_dim) - 1
#         self.layers_dim = layers_dim
#         self.device = device
#         self.init_weights()
#
#     def init_weights(self):
#         """ Initialize weights and biases of network based on layers dimension.
#             Store weights and biases in self.params.
#             weights and biases key should be of format "W#" and "b#" where # is the layer number.
#             Example: for layer 1, weight and bias key is "W1" and "b1"
#         Args:
#             None
#
#         Returns:
#             None
#         """
#         ### TODO: Initialize weights and bias of network
#         ### TODO: Store weights and biases in self.params
#         self.params = {
#             'W1': torch.randn((784, 512), requires_grad=True, device="cuda"),
#             'b1': torch.zeros((1, 512), requires_grad=True, device="cuda"),
#             'W2': torch.randn((512, 512), requires_grad=True, device="cuda"),
#             'b2': torch.zeros((1, 512), requires_grad=True, device="cuda"),
#             'W3': torch.randn((512, 10), requires_grad=True, device="cuda"),
#             'b3': torch.zeros((1, 10), requires_grad=True, device="cuda")
#         }
#         ### HINT: Remember to set require_grad to True
#         ### HINT: Remember to put tensors of target device
#
#     def forward(self, x):
#         """ Perform forward pass
#         Args:
#             x (torch.tensor): tensor of shape of (B, C, H, W)
#
#         Return:
#             torch.tensor: tensor of shape of (B, N_classes)
#         """
#         ### TODO: Fill out this function
#         x = torch.flatten(x, start_dim=1)
#         x = torch.mm(x, self.params['W1'])
#         x = x + self.params['b1']
#         x = Sigmoid(x)
#
#         x = torch.mm(x, self.params['W2'])
#         x = x + self.params['b2']
#         x = Sigmoid(x)
#
#         x = torch.mm(x, self.params['W3'])
#         x = x + self.params['b3']
#         x = Sigmoid(x)
#
#         return x
#
def calc_accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    return (classes == labels).float().sum()

def Train(model, optimizer, dataloader, device):
    """ performs training on train set
    Args:
        model (my_nn instance): model to be trained
        optimizer (torch.optim instance)
        dataloader (torch.utils.data.DataLoader instance): dataloader for train set
        device (str): computation device ['cpu','cuda',...]
    Returns:
        list of floats: mini_batch loss sampled every 20 steps for visualization purposes
        list of floats: mini_batch accuracy sampled every 20 steps for visualization purposes
    """
    loss_tracker = []
    accuracy_tracker = []
    for i, (data, label) in enumerate(dataloader):
        ### TODO: Put data and label on target device
        data = data.to(device).to(torch.float32)
        labels = label.to(device)
        ### TODO: Set gradients to zero
        optimizer.zero_grad()
        ### TODO: Pass data to the model
        out = model.forward(data)
        ### TODO: Calculate the loss of predicted labels vs ground truth labels
        loss = CE_loss(out, labels)
        ### TODO: Calculate gradients and update weights and biases
        loss.backward()
        optimizer.step()

        if i % 20:
            with torch.no_grad():
                loss_tracker.append(loss.item())
                ### TODO: calculate accuracy of mini_batch
                accuracy = calc_accuracy(out, labels)
                accuracy_tracker.append(accuracy / data.size(0))

    return loss_tracker, accuracy_tracker

#
def Test(model, dataloader, device):
    """ performs training on train set
    Args:
        model (my_nn instance): model to be trained
        dataloader (torch.utils.data.DataLoader instance)
        device (str): computation device ['cpu','cuda',...]
    Returns:
        floats: test set loss for visualization purposes
        floats: test set accuracy for visualization purposes
    """
    loss_tracker = []
    accuracy_tracker = []
    for i, (data, label) in enumerate(dataloader):
        ### TODO: Put data and label on target device
        data = data.to(device).to(torch.float32)
        labels = label.to(device)

        with torch.no_grad():
            ### TODO: Pass data to the model
            out = model.forward(data)
            ### TODO: Calculate the loss of predicted labels vs ground truth labels
            loss = CE_loss(out, labels)

            ### TODO: calculate accuracy of mini_batch
            accuracy = calc_accuracy(out, labels)

        loss_tracker.append(loss.item())
        accuracy_tracker.append(accuracy / data.size(0))

    return sum(loss_tracker) / len(loss_tracker), sum(accuracy_tracker) / len(accuracy_tracker)
#
#
# # # Training hyper parameters
# # epochs = 10
# # learning_rate = 0.001
# # layers_dim = [28*28,512,512,10]
# #
# # ### TODO: Set target device for computations
# # device = 'cuda'
# # print(f'device: {device}')
# #
# # ### TODO: Initialize model using layers_dim
# # model = my_nn(layers_dim, layers_activation='sigmoid', device=device)
# #
# # ### TODO: Initialize Adam optimizer
# # optimizer = torch.optim.Adam(model.params.values(), lr=learning_rate)
# #
# # train_loss_tracker = []
# # train_accuracy_tracker = []
# #
# # test_loss_tracker = []
# # test_accuracy_tracker = []
# #
# # for epoch in range(epochs):
# #     print(f'Epoch: {epoch}')
# #     train_loss,train_accuracy = Train(model,optimizer,kmnist_train_loader,device)
# #     test_loss , test_accuracy = Test(model,kmnist_test_loader,device)
# #     train_loss_tracker.extend(train_loss)
# #     train_accuracy_tracker.extend(train_accuracy)
# #     test_loss_tracker.append(test_loss)
# #     test_accuracy_tracker.append(test_accuracy)
# #     print('\t training loss/accuracy: {0:.2f}/{1:.2f}'.format(sum(train_loss)/len(train_loss), sum(train_accuracy)/len((train_accuracy))))
# #     print('\t testing loss/accuracy: {0:.2f}/{1:.2f}'.format(test_loss, test_accuracy))

# load dataset from path
# set path to images location on your local machine or google drive
path = './images'
dataset = ImageFolder(path)
print(f'number of images: {len(dataset)}')
print(f'number of classes: {len(dataset.classes)}')


# Create train and test splits of original dataset
test_pct = 0.3
test_size = int(len(dataset)*test_pct)
train_size = len(dataset) - test_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])
class DogBreedDataset(Dataset):

    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)
            return img, label
batch_size =64

#train set transforms
train_transform = transforms.Compose([
   transforms.Resize((240, 240)),
    transforms.ToTensor()
])

# test set transforms
test_transform = transforms.Compose([
    transforms.Resize((240,240)),
    transforms.ToTensor()
])

# Initialize train and test sets
train_dataset = DogBreedDataset(train_ds, train_transform)
test_dataset = DogBreedDataset(test_ds, test_transform)

# Create DataLoaders
train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size)

class conv_net(nn.Module):
    def __init__(self):
        """ Initialize conv_net
        Args:
            None
        Returns:
            None
        """
        super(conv_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=100, kernel_size=11, stride=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=512, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.num_features = 256 * 2 * 2
        self.linear = nn.Linear(in_features=self.num_features, out_features=120)
        self.init_weights()

    def init_weights(self):
        ### EXTRA CREDIT FOR ALL: initialize network weights based on Xavier initialization
        torch.nn.Parameter(nn.init.xavier_uniform_(self.conv1.weight))
        torch.nn.Parameter(nn.init.xavier_uniform_(self.conv2.weight))
        torch.nn.Parameter(nn.init.xavier_uniform_(self.conv3.weight))
        torch.nn.Parameter(nn.init.xavier_uniform_(self.conv4.weight))
        torch.nn.Parameter(nn.init.xavier_uniform_(self.conv5.weight))
        torch.nn.Parameter(nn.init.xavier_uniform_(self.linear.weight))

    def forward(self, x):
        """ Perform forward pass
        Args:
            x (torch.tensor): tensor of images of shape  (B, C, H, W)
        Returns:
            torch.tensor: tesnor of output of shape (B, N_classes)
        """
        ### TODO: fill out this function
        activation = nn.LeakyReLU()
        x = activation(self.conv1(x))
        x = self.max_pool1(x)
        x = activation(self.conv2(x))
        x = activation(self.conv3(x))
        x = self.max_pool2(x)
        x = activation(self.conv4(x))
        x = activation(self.conv5(x))
        x = self.max_pool3(x)
        x = x.view(-1, self.num_features)
        x = activation(self.linear(x))
        return x

# Training hyper parameters
epochs = 20
learning_rate = 0.001

### TODO: Set target device for computations
device = 'cuda'
print(f'device: {device}')


### TODO: Initialize conv_net
model = conv_net().cuda()
### TODO: Put model parameters on target device
model.to(device)
### TODO: Initialize Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss_tracker = []
train_accuracy_tracker = []

test_loss_tracker = []
test_accuracy_tracker = []

for epoch in range(epochs):
    train_loss,train_accuracy = Train(model,optimizer,train_dl,device)
    test_loss , test_accuracy = Test(model,test_dl,device)
    train_loss_tracker.extend(train_loss)
    train_accuracy_tracker.extend(train_accuracy)
    test_loss_tracker.append(test_loss)
    test_accuracy_tracker.append(test_accuracy)
    print(f'epoch: {epoch}')
    print('\t training loss/accuracy: {0:.2f}/{1:.2f}'.format(sum(train_loss)/len(train_loss), sum(train_accuracy)/len((train_accuracy))))
    print('\t testing loss/accuracy: {0:.2f}/{1:.2f}'.format(test_loss, test_accuracy))
