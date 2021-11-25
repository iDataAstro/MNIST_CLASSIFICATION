import tensorflow as tf
import logging
import numpy as np
from typing import Tuple


def get_prepared_data(validation_datasize: int) -> \
        Tuple[Tuple[np.uint64, np.uint64], Tuple[np.uint64, np.uint64], Tuple[np.uint64, np.uint64]]:
    """Function will return MNIST data
    Args:
        validation_datasize: validation datasize for split data
    Returns:
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
    """

    logging.info("Downloading data..")
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    logging.info("Normalize and split data..")
    X_valid, X_train = X_train_full[:validation_datasize]/255., X_train_full[validation_datasize:]/255.
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:]
    X_test = X_test/255.
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
