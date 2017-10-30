from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import matplotlib.pyplot as plt

def visualize(embed, labels):

    labelset = set(labels.tolist())

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    #fig, ax = plt.subplots()
    for label in labelset:
        indices = np.where(labels == label)
        ax.scatter(embed[indices,0], embed[indices,1], label = label, s = 20)
    ax.legend()
    fig.savefig('embed.jpeg', format='jpeg', dpi=600, bbox_inches='tight')
    plt.show()

mnist = input_data.read_data_sets('MNIST_data', one_hot = False)

mnist_test_labels = mnist.test.labels

embed = np.fromfile('embed.txt', dtype = np.float32)
embed = embed.reshape([-1, 2])

visualize(embed, mnist_test_labels)
