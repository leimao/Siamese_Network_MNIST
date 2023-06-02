import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

def visualize(embed, labels, tag, results_dir='./'):
    labelset = set(labels.tolist())

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    for label in labelset:
        indices = np.where(labels == label)
        ax.scatter(embed[indices,0], embed[indices,1], label = label, s = 5)
    ax.legend()
    fig.savefig(f'{results_dir}{os.sep}embed_{tag}.jpeg', format='jpeg', dpi=600, bbox_inches='tight')
    plt.close()

# # Load the MNIST dataset
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# # Get the labels for the test data
# mnist_test_labels = mnist.targets.numpy()

# # Load embedding from file
# file_name = 'embed_ep:60.pt'
# embed = torch.load(file_name, map_location=torch.device('cpu')).numpy()
# embed = embed.reshape([-1, 2])

# tag = file_name.split('.')[0].split('_')[-1]

# visualize(embed, mnist_test_labels, tag)