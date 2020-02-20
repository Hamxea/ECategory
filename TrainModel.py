import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split


class TrainModel():
    """ Train model class """

    def get_fashion_mnist_data(self):
        """ get mnist dataset """
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        return (x_train, y_train), (x_test, y_test)

    def train_val_data_split(self, x_train, y_train):
        """Here we split validation data to optimiza classifier during training """
        X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=13)

        return (X_train, y_train), (X_val, y_val)

    def fashion_mnist_split(self):
        """ split the dataset inot train, validation and test data """
        (x_train, y_train), (x_test, y_test) = self.get_fashion_mnist_data()
        (x_train, y_train), (x_val, y_val) = self.train_val_data_split(x_train=x_train, y_train=y_train)

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def train(self):
        """ train model"""

        # get data splits
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.fashion_mnist_split()

        batch_size = 256
        num_classes = 10
        epochs = 50

        # input image dimensions
        img_rows, img_cols = 28, 28
        input_shape = (img_rows, img_cols, 1)

        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_val = X_val.astype('float32')
        X_train /= 255
        X_test /= 255
        X_val /= 255

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         kernel_initializer='he_normal',
                         input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        print(model.summary())

        history = model.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(X_val, y_val))
        score = model.evaluate(X_test, y_test, verbose=0)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        return (score[1], score[0])
