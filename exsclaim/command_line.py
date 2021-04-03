import argparse
import pathlib
import os
import json
from exsclaim.pipeline import Pipeline
from exsclaim.figures.scale.train_label_reader import train_crnn
from exsclaim.figures.scale.evaluate_scale import test_label_reading


def run_pipeline():
    # Parse Command Line arguments, if present
    parser = argparse.ArgumentParser(
        description=(
            "Automatic EXtraction, Separation, and Caption-based natural "
            "Language Annotation of IMages from scientific figures"
        )
    )
    parser.add_argument(
        'query', 
        type=str, default=None,
        help=(
            'Path to EXSCLAIM Query JSON, defined here: '
            'https://github.com/MaterialEyes/exsclaim/wiki/JSON-Schema#query-json-'
            '. Samples are in the query folder.'
        )
    )
    parser.add_argument(
        '--tools', '-t',
        type=str, default='jcf',
        help=(
            'String containing the first letter of each tool to be run on'
            ' input query.\nJ\tJournalScraper\nC\tCaptionDistributor\n'
            'F\FigureSeparator. Order and case insensitive. If no value'
            ' is supplied, all are run'
        )
    )
    args = parser.parse_args()
    # Format args to run enter into Pipeline
    tools = args.tools.lower()
    f = "f" in tools
    j = "j" in tools
    c = "c" in tools
    # Run the pipeline
    pipeline = Pipeline(args.query)
    pipeline.run(journal_scraper=j, caption_distributor=c, figure_separator=f)

def train():
    current_file = pathlib.Path(__file__).resolve(strict=True)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m", "--model",
        type=str,
        help=(
            "Name of model you wish to train (i.e. scale_label_reader,"
            " scale_object_detector, etc.). Configuration file will be "
            "assumed to be exsclaim/figures/config/<model>.json if '.json' "
            "extenstion is not supplied."
        )
    )
    ap.add_argument(
        "-n", "--name",
        type=str,
        help=(
            "Name of the configuration to train. The arguments in <model>.json"
            " corresponding to the <name> key will be used to train."
        )
    )
    ap.add_argument(
        "-t", "--test",
        default=False, action="store_true",
        help="Run the model on test images instead of training."
    )
    args = ap.parse_args()

    # # Load configuration
    # training_directory = parent_directory / "training"
    if ".json" not in args.model:
        model_config_path = (
            current_file / "figures" / "config" / (args.model + ".json")
        )
    else:
        model_config_path = args.model
    with open(model_config_path, "r") as f:
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
                convolution_layers = config["convolution_layers"]
            )


def activate_ui():
    current_file = pathlib.Path(__file__).resolve(strict=True)
    ui_dir = current_file.parent / "ui"
    os.chdir(ui_dir)
    print(os.getcwd())
    parser = argparse.ArgumentParser(
        description=(
            'Run the EXSCLAIM! Pipeline User Interface. This command will'
            ' tell Django to start serving the interface to localhost. '
            'To use the interface, open http://127.0.0.1:8000/ in a browser'
        )
    )
    ## copied fron Django manage.py, replacing sysarg with 'runserver'
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'exsclaim.ui.exsclaim_gui.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    fake_sysargv = ["manage.py", "runserver"]
    execute_from_command_line(fake_sysargv)