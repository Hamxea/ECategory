from __future__ import print_function

from keras.datasets import fashion_mnist

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class FashionPlot():
    """ """

    def get_fashion_mnist_data(self):
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

        return (X_train, y_train), (X_test, y_test)

    def fashion_plot(self):

        (X_train, y_train), (X_test, y_test) = self.get_fashion_mnist_data()

        fashion_name = ['T-shirt/top',
                        'Trouser',
                        'Pullover',
                        'Dress',
                        'Coat',
                        'Sandal',
                        'Shirt',
                        'Sneaker',
                        'Bag',
                        'Ankle boot']

        labels = pd.DataFrame(columns=['index', 'fname', 'label', 'real_int', 'real_str'])

        for i in range(0, 10000):
            fig = plt.figure(facecolor='white')

            labels_in_picture = [0] * 10
            # print(labels_in_picture)

            real_int = ''
            real_str = ''
            for j in range(1, 5):
                ax = plt.subplot(1, 4, j)
                ax.set_axis_off()
                idx = np.random.randint(0, 1000)
                plt.imshow(X_train[idx], cmap=plt.get_cmap('binary'))
                labels_in_picture[y_train[idx]] = 1

                real_int += str(y_train[idx])
                if j != 4:
                    real_int += ', '

                real_str += str(fashion_name[y_train[idx]])
                if j != 4:
                    real_str += ', '

            print(labels_in_picture)

            fig.subplots_adjust(wspace=0, hspace=0)

            fname = "./images_dataset/" + "%d" % i + ".png"
            plt.savefig(fname, bbox_inches='tight', pad_inches=0)
            plt.clf()

            # plt.show()

            labels = labels.append({"index": i,
                                    "fname": fname,
                                    "label": labels_in_picture,
                                    "real_int": real_int,
                                    "real_str": real_str
                                    }, ignore_index=True)

        labels.to_csv("./data/labels2.csv", index=False)
