from src.utils.common import read_config, get_unique_name
from src.utils.data_management import get_prepared_data
from src.utils.model import get_prepared_model, save_model, save_history_plot, get_callbacks

import argparse
import logging
import os


logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)-20s]: %(message)s",
    filemode="a"
)


def training(config_path: str, stage: str) -> None:
    """
    Function create, compile, train and save ANN model.
    Args:
        config_path: Path for config file.
        stage ([str]): stage of experiment
    """
    logging.info("=" * 50)
    configs = read_config(config_path)

    validation_datasize = configs["params"]["validation_datasize"]
    no_classes = configs["params"]["no_classes"]
    input_shape = configs["params"]["input_shape"]
    loss = configs["params"]["loss_function"]
    optimizer = configs["params"]["optimizer"]
    metrics = configs["params"]["metrics"]
    checkpoint_dir = configs["artifacts"]["checkpoint_dir"]
    tensorboard_logs = configs["logs"]["tensorboard_logs"]
    EPOCHS = configs["params"]["epochs"]
    plot_dir = configs["artifacts"]["plot_dir"]
    model_dir = configs["artifacts"]["model_dir"]

    logging.info("Getting data..")
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_prepared_data(validation_datasize)

    if stage == 'ALL':
        for in_stage in ["BASE_MODEL", "KERNEL_INIT_MODEL", "BN_BEFORE_MODEL", "BN_AFTER_MODEL"]:
            logging.info("-" * 50)
            logging.info(f"Training Started for {in_stage}..")
            logging.info(f"Getting compiled ann model for {in_stage}..")

            model_ann = get_prepared_model(in_stage, no_classes, input_shape, loss, optimizer, metrics)
            model_ann.summary(print_fn=logging.info)
            callback_list = get_callbacks(checkpoint_dir, tensorboard_logs, in_stage)
            logging.info(f"Model training start for {in_stage}..")
            history = model_ann.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_valid, y_valid),
                                    callbacks=callback_list)
            logging.info(f"Model training ends for {in_stage}..")
            logging.info(f"Plot Loss/Accuracy curves for {in_stage} ..")
            save_history_plot(history, plot_dir, in_stage)

            model_suffix = get_unique_name(in_stage)
            save_model(model_dir, model_ann, model_suffix)
            logging.info(f"Model saved successfully for {in_stage}..")
            logging.info("-" * 50)
    else:
        logging.info("-" * 50)
        logging.info(f"Training Started for {stage} ..")
        logging.info(f"Getting compiled ann model for {stage}..")

        model_ann = get_prepared_model(stage, no_classes, input_shape, loss, optimizer, metrics)
        model_ann.summary(print_fn=logging.info)
        callback_list = get_callbacks(checkpoint_dir, tensorboard_logs, stage)
        logging.info(f"Model training start for {stage}..")
        history = model_ann.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_valid, y_valid),
                                callbacks=callback_list)
        logging.info(f"Model training ends for {stage} ..")
        logging.info(f"Plot Loss/Accuracy curves for {stage}..")
        save_history_plot(history, plot_dir, stage)

        model_suffix = get_unique_name(stage)
        save_model(model_dir, model_ann, model_suffix)
        logging.info(f"Model saved successfully for {stage}..")
        logging.info("-" * 50)
    logging.info("=" * 50)


if __name__ == '__main__':
    args = argparse.ArgumentParser(prog="training.py",
                                   usage='%(prog)s --config|-c config_file_path',
                                   description="To train model provide configuration file and stage")

    args.add_argument("--config", "-c", default="config.yaml")
    args.add_argument("--stage", "-s", default="ALL", help="[ALL | BASE_MODEL | KERNEL_INIT_MODEL | BN_BEFORE_MODEL \
                                                           | BN_AFTER_MODEL]")

    parsed_args = args.parse_args()

    training(parsed_args.config, parsed_args.stage)
