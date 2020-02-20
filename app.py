from flask import Flask

from TrainModel import TrainModel

app = Flask(__name__)

train_model = TrainModel()


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/train', methods=['GET', 'POST'])
def train():
    model = train_model.train()
    return


if __name__ == '__main__':
    app.run()
