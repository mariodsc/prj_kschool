"""A simple main file to showcase the template."""

import logging.config
import argparse

from tensorflow.keras import models, layers, activations, datasets
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras.utils import to_categorical

LOGGER = logging.getLogger()

def _download_data():
    LOGGER.info('Donwload data')
    train, test = datasets.mnist.load_data()
    x_train, y_train = train
    x_test, y_test = test
    return x_train, y_train, x_test, y_test


def _preprocess_data(x, y):
    LOGGER.info('Prepro')
    x = x / 255.0
    y = to_categorical(y)
    return x,y

def _build_model():
    LOGGER.info('Build model')
    m = models.Sequential()
    m.add(layers.Input((28,28), name='my_input_layer'))
    m.add(layers.Flatten())
    m.add(layers.Dense(128, activation=activations.relu))
    m.add(layers.Dense(64, activation=activations.relu))
    m.add(layers.Dense(32, activation=activations.relu))
    m.add(layers.Dense(10, activation=activations.softmax))
    return m

def train_and_evaluate(batch_size, epochs, job_dir, output_path):
    # Donwload the data
    x_train, y_train, x_test, y_test = _download_data()

    # Preprocess the data
    x_train, y_train = _preprocess_data(x_train, y_train)
    x_test, y_test = _preprocess_data(x_test, y_test)

    # Build the model
    model = _build_model()
    model.compile(loss=losses.categorical_crossentropy
          , optimizer=optimizers.Adam()
          , metrics=[metrics.categorical_accuracy])

    # Train the model
    model.fit(x_train, y_train
        , batch_size=batch_size
        , epochs=epochs)
      #  , validation_split=0.15) no se utiliza

    # Evaluate the model
    loss_value, accuracy = model.evaluate(x_test, y_test)
    #print(accuracy) -- se puede hacer pero mejor logger
    LOGGER.info(f" **** LOSS VALUE:{loss_value}, ACCURACY:{round(accuracy,4)}")
    

def main():
    """Entry point for your module."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, help='Batch size for the training')
    parser.add_argument('--epochs', type=int, help='Batch size for the training')
    parser.add_argument('--job-dir', default=None, required=False, help='Option for AI platform')
    parser.add_argument('--model-ouput-path', help='Path to write the SaveModel format')

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    job_dir = args.job_dir
    output_path = args.model_output_path

    train_and_evaluate(batch_size, epochs, job_dir, output_path)



if __name__ == "__main__":
    main()
