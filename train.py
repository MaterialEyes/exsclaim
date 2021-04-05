import pathlib
import json
import argparse

from exsclaim.figures.scale.train_label_reader import train_crnn
from exsclaim.figures.scale.evaluate_scale import test_label_reading


current_file = pathlib.Path(__file__).resolve(strict=True)
parent_directory = current_file.parent

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
    help=("Name of model you wish to train (i.e. scale_label_reader,"
    " scale_object_detector, etc.). Configuration file will be assumed to"
    " be training/<model>.json"))
ap.add_argument("-n", "--name", type=str,
    help=("Name of the configuration to train. The arguments in <model>.json"
    " corresponding to the <name> key will be used to train."))
ap.add_argument("-t", "--test", default=False, action="store_true",
    help="Run the model on test images instead of training.")

args = ap.parse_args()

# Load configuration
training_directory = parent_directory / "training"
with open(training_directory / (args.model + ".json"), "r") as f:
    configuration_dict = json.load(f)
config = configuration_dict[args.name]

if args.model == "scale_label_reader":
    if args.test:
        test_label_reading(args.name)
    else:
        train_crnn(batch_size = config["batch_size"],
            learning_rate = config["learning_rate"],
            cnn_to_rnn = config["cnn_to_rnn"],
            model_name = args.name,
            input_height = config["input_height"],
            input_width = config["input_width"],
            sequence_length = config["sequence_length"],
            recurrent_type = config["recurrent_type"],
            cnn_kernel_size = config["cnn_kernel_size"],
            convolution_layers = config["convolution_layers"],
            text = config.get("text", "random_separate")
        )
