import torch
import os
import sys; sys.path.append('./')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from siamese_simple import *
from visualize import visualize
from torchsummary import summary
from time import time

# set plot params to use latex
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

# Define constants
EPISODE_MAX = 50
BATCH_SIZE = 128
GRAD_CLIP_NORM = 5

# Preparing MNIST dataset
mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data/', train=True, transform=mnist_transform, download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=mnist_transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the learning rates for testing
learning_rates = [0.001, 0.01, 0.1, 1]

# loop over learning rates
for LEARNING_RATE in learning_rates:
    # print the 

    # create the directory structure for storing the results
    model_dir = f'./sandbox/experiment-learn-rate-may-30/learning_rate_{LEARNING_RATE}/model'
    results_dir = f'./sandbox/experiment-learn-rate-may-30/learning_rate_{LEARNING_RATE}/results'
    logs_dir = f'./sandbox/experiment-learn-rate-may-30/learning_rate_{LEARNING_RATE}/logs'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

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
            torch.nn.utils.clip_grad_norm_(siamese_net.parameters(), max_norm=GRAD_CLIP_NORM)
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
            torch.save(siamese_net.state_dict(), f'{model_dir}{os.sep}siamese_best.pt')

        # Save model outputs
        if (episode+1) % 10 == 0:
            # save the loss history
            torch.save(losses,f'{logs_dir}{os.sep}loss_history.pt')

            # save embeddings to text file
            save_embeddings_cnn(siamese_net, test_loader, filename=f'{logs_dir}{os.sep}embed_ep:{episode+1}.pt')

            # load the embeddings and save visualize
            embed = torch.load(f'{logs_dir}{os.sep}embed_ep:{episode+1}.pt',map_location=torch.device('cpu')).numpy()
            embed = embed.reshape([-1,2])
            tag = str(episode + 1)
            test_labels = test_dataset.targets.numpy()
            visualize(embed, test_labels, tag, results_dir)

            # create a 2 row x 1 column figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

            # plot the training loss on ax1
            ax1.plot(losses['train'], label='train')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')

            # plot the test loss on ax2
            ax2.plot(losses['test'], label='test')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Loss')
            ax2.set_title('Test Loss')

            # save the figure
            plt.savefig(f'{results_dir}{os.sep}loss_history.png', dpi=300)