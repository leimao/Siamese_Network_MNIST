'''
Siamese Network Implementation Practice
Lei Mao
10/13/2017
University of Chicago
'''

'''
References
TesorFlow Sharing Variables
https://www.tensorflow.org/versions/r0.12/how_tos/variable_scope/
Simple Siamese Network
https://github.com/ywpkwon/siamese_tf_mnist/blob/master/inference.py
'''
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random

LEARNING_RATE = 0.001
SAVE_PERIOD = 500
MODEL_DIR = 'model/' # path for saving the model
MODEL_NAME = 'siamese_model.pt'
RAND_SEED = 0 # random seed
# tf.set_random_seed(RAND_SEED)

class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )
        # # Initialize
        # # First input image
        # self.tf_input_1 = tf.placeholder(tf.float32, [None, 784], name = 'input_1')
        # # Second input image
        # self.tf_input_2 = tf.placeholder(tf.float32, [None, 784], name = 'input_2')
        # # Label of the image pair
        # # 1: paired, 0: unpaired
        # self.tf_label = tf.placeholder(tf.float32, [None,], name = 'label')
        # # Output
        # self.output_1, self.output_2 = self.network_initializer()
        # # Loss
        # self.loss = self.loss_contrastive()
        # # Optimizer
        # self.optimizer = self.optimizer_initializer()
        # # Initialize tensorflow session
        # self.saver = tf.train.Saver()
        # self.sess = tf.Session()
        # self.sess.run(tf.global_variables_initializer())
    
    def forward_once(self, x):
        return self.fc_layer(x)
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class Siamese_CNN(nn.Module):

    def __init__(self):
        super(Siamese_CNN, self).__init__()

        # Define the architecture of the sub-networks
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(7*7*64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
        
    def forward_once(self, x):
        output = self.conv_layer(x)
        output = output.view(output.size()[0], -1)  # Flatten the output
        output = self.fc_layer(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    
class ContrastiveLoss(nn.Module):

    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        eucliden_distance = (output1 - output2).pow(2).sum(1)
        loss = torch.mean((label) * eucliden_distance + (1 - label) * torch.pow(torch.clamp(self.margin - eucliden_distance.sqrt(), min=0.0), 2))
        return loss
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

siamese_net = Siamese().to(device)
criterion = ContrastiveLoss()
optimizer = optim.SGD(siamese_net.parameters(), lr=LEARNING_RATE)

def save_model(model_name):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(siamese_net.state_dict(), MODEL_DIR + model_name)

def load_model():
    siamese_net.load_state_dict(torch.load(MODEL_DIR + MODEL_NAME))

# Function for computing test loss
def compute_test_loss(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0

    with torch.no_grad():  # No need to track gradients in testing
        for i, (images, labels) in enumerate(test_loader):
            images = images.view(-1, 28*28).to(device)
            labels = labels.to(device)
            pairs, targets = create_pairs(images, labels)
            
            # Forward pass
            output1, output2 = model(pairs[:, 0], pairs[:, 1])
            loss = criterion(output1.to(device), output2.to(device), targets.to(device))

            total_loss += loss.item()

    # Return average test loss
    return total_loss / len(test_loader)

# Function for computing test loss
def compute_test_loss_cnn(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0

    with torch.no_grad():  # No need to track gradients in testing
        for i, (images, labels) in enumerate(test_loader):
            images = images.view(-1,1,28,28).to(device)
            labels = labels.to(device)
            pairs, targets = create_pairs(images, labels)
            
            # Forward pass
            output1, output2 = model(pairs[:, 0], pairs[:, 1])
            loss = criterion(output1.to(device), output2.to(device), targets.to(device))

            total_loss += loss.item()

    # Return average test loss
    return total_loss / len(test_loader)


# Create a balanced set of pairs for each batch
def create_pairs(images, labels):
    digit_indices = [torch.where(labels == i)[0] for i in range(10)]
    pairs = []
    labels = []

    # find where the length of each element in digit indices is nonzero
    good_label_digits = np.where([len(d) > 0 for d in digit_indices])[0]

    # redefine the digit_indices
    digit_indices = [digit_indices[i] for i in good_label_digits]

    for d in good_label_digits:
        for i in range(len(digit_indices[d]) - 1):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs.append(torch.stack([images[z1], images[z2]]))  # change here
            inc = random.randrange(1, len(good_label_digits))
            dn = (d + inc) % len(good_label_digits)
            j = (i + inc) % len(digit_indices[dn])
            z1, z2 = digit_indices[d][i], digit_indices[dn][j]
            pairs.append(torch.stack([images[z1], images[z2]]))
            labels.extend([1, 0])
            
    return torch.stack(pairs), torch.tensor(labels)

def save_embeddings(model, loader, filename='embed.txt'):
    all_embeddings = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.view(-1, 28*28).to(device)
            embeddings = model.forward_once(images)  # Assuming your network's forward method takes one argument
            all_embeddings.append(embeddings)

    # convert to torch tensor
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # save to file
    torch.save(all_embeddings, filename)

def save_embeddings_cnn(model, loader, filename='embed.txt'):
    all_embeddings = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.view(-1,1,28,28).to(device)
            embeddings = model.forward_once(images)  # Assuming your network's forward method takes one argument
            all_embeddings.append(embeddings)

    # convert to torch tensor
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # save to file
    torch.save(all_embeddings, filename)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)