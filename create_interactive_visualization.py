import plotly.subplots as sp
import plotly.graph_objs as go
import sys; sys.path.append('./')
import torch
import numpy as np
from visualize import visualize
from torchvision import datasets, transforms

# Preparing MNIST dataset
mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_dataset = datasets.MNIST(root='./data/', train=False, transform=mnist_transform, download=True)
test_labels = test_dataset.targets.numpy()

# create a 1x1 subplot
fig = sp.make_subplots(rows=1, cols=1)

# get the embedding results
embed = torch.load("embed_ep:50.pt", map_location=torch.device('cpu')).numpy()
embed = embed.reshape([-1, 2])

labelset = set(test_labels.tolist())
for label in labelset:
    indices = np.where(test_labels == label)

    # binarize the image data and format it into a string for the hovertext
    image_data = []
    for d in indices[0]:
        binary_image = (test_dataset[d][0][0].numpy() > 0.5)*1
        text_image = ""
        for row in binary_image:
            text_row = ''.join(['.' if pixel == 0 else '#' for pixel in row])
            text_image += text_row + '<br>'
        text_image += ""
        image_data.append(text_image)    
    
    fig.add_trace(
        go.Scatter(
            x=embed[indices,0][0],
            y=embed[indices,1][0],
            mode='markers',
            name=str(label),
            hovertext=image_data,
            hoverinfo='text',
        ),
        row=1,
        col=1,
    )

# Set x and y axis limits
fig.update_xaxes(range=[-10, 10], row=1, col=1)
fig.update_yaxes(range=[-10, 10], row=1, col=1)

fig.update_layout(
    hoverlabel=dict(
        font_size=4,
        font_family="Courier New",
    )
)

# save the figure as an image
fig.write_image('interactiveScatterPlot.jpeg', width=800, height=800, scale=2)

# save the figure as an html file
fig.write_html('index.html')