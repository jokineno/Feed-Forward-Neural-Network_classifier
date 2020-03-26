import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

RESOLN = (28,28)

def plot_mnist_weights(weights, label):
    fig = plt.figure(figsize=(16,7))
    for i in range(10):
        img = np.reshape(weights[i,:], RESOLN)
        plt.subplot(2,5,i+1)
        plt.imshow(img, cmap=cm.gray)
    fig.savefig('weights_' + label + '.png')


def plot_accs(train_accs, test_accs, epochs):
    OPT_ORDER = ['NO_OPT', 'ADAM', 'MOMENT', 'NO_REG', 'DROP', 'L2']
    for i, option in enumerate(OPT_ORDER): #Draw the plots in the order of OPT_ORDER
        if option not in train_accs: #But plot an option only if the the user has supplied its values
            continue
        fig = plt.figure(figsize=(16,4))
        plt.plot(epochs[option], train_accs[option], linestyle=':', marker='d')
        plt.plot(epochs[option], test_accs[option], linestyle='-', marker='o')
        plt.ylim([50, 100])
        plt.xticks(epochs[option])
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.title(option)
        plt.legend(['train acc', 'test acc'])
        plt.grid(True)
        fig.savefig('accs_' + option + '.png')
        