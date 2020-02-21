from flask import Flask

from FashionPlot import FashionPlot
from TrainModel import TrainModel

app = Flask(__name__)

train_model = TrainModel()
fashion = FashionPlot()


@app.route('/')
def e_category():
    return 'ECategory!'


@app.route('/train', methods=['GET', 'POST'])
def train():
    model = train_model.train()
    return "model trained successful"


@app.route('/fashionPlot', methods=['GET', 'POST'])
def fashion_plot():
    fashion_plot = fashion.fashion_plot()
    return "fashion plotted successful"


if __name__ == '__main__':
    app.run()
