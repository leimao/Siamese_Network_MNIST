from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from siamese_simple import *
import torch
import matplotlib.pyplot as plt

EPISODE_MAX = 50000
BATCH_SIZE = 128

# Preparing MNIST dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
siamese_net = Siamese().to(device)
criterion = ContrastiveLoss()
optimizer = torch.optim.SGD(siamese_net.parameters(), lr=LEARNING_RATE)

# Dictionary for storing losses
losses = {
    'train': [],
    'test': [],
}

# Adjusted training loop with loss storage
for episode in range(EPISODE_MAX):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)
        pairs, targets = create_pairs(images, labels)

        # Forward pass
        output1, output2 = siamese_net(pairs[:, 0], pairs[:, 1])
        loss = criterion(output1, output2, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 20 == 0:
            print('Episode [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(episode+1, EPISODE_MAX, i+1, len(train_loader), loss.item()))

    # Store training loss for this episode
    losses['train'].append(loss.item())

    # Compute test loss
    test_loss = compute_test_loss(siamese_net, test_loader, criterion, device)
    losses['test'].append(test_loss)

    print('Episode [{}/{}], Test Loss: {:.4f}'.format(episode+1, EPISODE_MAX, test_loss))

# Save model
if (episode+1) % 1000 == 0:
    save_model()