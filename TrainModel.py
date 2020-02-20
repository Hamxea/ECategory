from keras.datasets import fashion_mnist


class TrainModel():

    def fashion_mnist_split(self):
        """ get mnist dataset """

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        return (x_train, y_train), (x_test, y_test)

    def train(self):
        """ train model"""

        (x_train, y_train), (x_test, y_test) = self.fashion_mnist_split()

        return