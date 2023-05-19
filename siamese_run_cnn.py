from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from siamese_simple import *
from torchsummary import summary
import torch
import sys
import matplotlib.pyplot as plt

EPISODE_MAX = 50000
BATCH_SIZE = 128

# Preparing MNIST dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
siamese_net = Siamese_CNN().to(device)
summary(siamese_net, input_size=[(1, 28, 28), (1, 28, 28)])
criterion = ContrastiveLoss()
optimizer = torch.optim.SGD(siamese_net.parameters(), lr=LEARNING_RATE)

# Dictionary for storing losses
losses = {
    'train': [],
    'test': [],
}

# Initialize best test loss to a high value
best_test_loss = float('inf')

# Adjusted training loop with loss storage
for episode in range(EPISODE_MAX):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 1, 28, 28).to(device)
        labels = labels.to(device)
        pairs, targets = create_pairs(images, labels)

        # Forward pass
        output1, output2 = siamese_net(pairs[:, 0], pairs[:, 1])
        loss = criterion(output1.to(device), output2.to(device), targets.to(device))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(siamese_net.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (i+1) % 20 == 0:
            sys.stdout.write('\rEpisode [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(episode+1, EPISODE_MAX, i+1, len(train_loader), loss.item()))
            sys.stdout.flush()

    # Store training loss for this episode
    losses['train'].append(loss.item())

    # Compute test loss
    test_loss = compute_test_loss_cnn(siamese_net, test_loader, criterion, device)
    losses['test'].append(test_loss)

    sys.stdout.write('\rEpisode [{}/{}], Test Loss: {:.4f}\n'.format(episode+1, EPISODE_MAX, test_loss))
    sys.stdout.flush()

    # Check if this model is better (i.e., has lower test loss)
    if test_loss < best_test_loss:
        print('Test loss improved from {:.4f} to {:.4f}, saving best model to siamese_best.pt'.format(best_test_loss, test_loss))
        best_test_loss = test_loss
        torch.save(siamese_net.state_dict(), 'siamese_best.pt')

    # Save model
    if (episode+1) % 10 == 0:
        # save the loss history
        torch.save(losses,f'loss_history.pt')

        # save embeddings to text file
        save_embeddings_cnn(siamese_net, test_loader, filename=f'embed_ep:{episode+1}.pt')