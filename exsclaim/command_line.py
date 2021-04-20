import argparse
import pathlib
import os
import json
import unittest
from exsclaim.pipeline import Pipeline
from exsclaim.figures.scale.train_label_reader import train_crnn
from exsclaim.figures.scale.evaluate_scale import test_label_reading
from exsclaim.utilities.postgres import initialize_database, modify_database_configuration


def main():
    current_file = pathlib.Path(__file__).resolve(strict=True)
    parent_directory = current_file.parent
    # Parse Command Line arguments, if present
    parser = argparse.ArgumentParser(
        description=(
            "Automatic EXtraction, Separation, and Caption-based natural "
            "Language Annotation of IMages from scientific figures"
        ),
        epilog=(
            ""
        )
    )
    subparsers = parser.add_subparsers(dest='command')

    # subparser for 'exsclaim run' command (running the pipeline)
    query_parser = subparsers.add_parser("run")
    query_parser.add_argument(
        'query', 
        type=str, default=None,
        help=(
            'Path to EXSCLAIM Query JSON, defined here: '
            'https://github.com/MaterialEyes/exsclaim/wiki/JSON-Schema#query-json-'
            '. Samples are in the query folder.'
        )
    )
    query_parser.add_argument(
        '--tools', '-t',
        type=str, default='jcf',
        help=(
            'String containing the first letter of each tool to be run on'
            ' input query.\nJ\tJournalScraper\nC\tCaptionDistributor\n'
            'F\FigureSeparator. Order and case insensitive. If no value'
            ' is supplied, all are run'
        )
    )
    # subparser for 'exsclaim train' command (training models)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "-m", "--model",
        type=str,
        help=(
            "Name of model you wish to train (i.e. scale_label_reader,"
            " scale_object_detector, etc.). Configuration file will be "
            "assumed to be exsclaim/figures/config/<model>.json if '.json' "
            "extenstion is not supplied."
        )
    )
    train_parser.add_argument(
        "-c", "--configuration",
        type=str,
        help=(
            "Name of the configuration to train. The arguments in <model>.json"
            " corresponding to the <configuration> key will be used to train."
        )
    )
    train_parser.add_argument(
        "-t", "--test",
        default=False, action="store_true",
        help="Run the model on test images instead of training."
    )
    # subparser for 'exsclaim view' command (starting django interface)
    view_parser = subparsers.add_parser("view",
        description=(
            'Run the EXSCLAIM! Pipeline User Interface. This command will'
            ' tell Django to start serving the interface to localhost. '
            'To use the interface, open http://127.0.0.1:8000/ in a browser'
        )
    )
    view_parser.add_argument(
        '--configuration', '-c', 
        type=str, default=str(current_file.parent / "database.ini"),
        help=(
            'Path to database configuration file. Should be a .ini file with '
            'login information for default postgres user (to create exsclaim '
            'user and database) and information for exisiting or desired '
            'exsclaim user'
        )
    )
    view_parser.add_argument(
        '--bind', '-b', 
        default=False, action="store_true",
        help=(
            'Bind input configuration to exsclaim installation. This will '
            'save the database configuration information to database.ini, '
            'saving it for future runs and making it the default.'
        )
    )
    view_parser.add_argument(
        '--initialize_postgres', '-i', 
        default=False, action="store_true",
        help=(
            'Initialize postgres setup by creating user from database.ini '
            'exsclaim field. Also create database "exsclaim".'
        )
    )
    # subparser for 'exsclaim test' command (running python unittests)
    view_parser = subparsers.add_parser("test",
        description=(
            'Test the exsclaim pipeline'
        )
    )

    args = parser.parse_args()
    if args.command == "run":
        # Format args to run enter into Pipeline
        tools = args.tools.lower()
        f = "f" in tools
        j = "j" in tools
        c = "c" in tools
        # Run the pipeline
        pipeline = Pipeline(args.query)
        pipeline.run(journal_scraper=j, caption_distributor=c, figure_separator=f)
    elif args.command == "train":
        # Load configuration
        training_directory = parent_directory / "training"
        if ".json" not in args.model:
            model_config_path = (
                current_file / "figures" / "config" / (args.model + ".json")
            )
        else:
            model_config_path = args.model
        with open(model_config_path, "r") as f:
            configuration_dict = json.load(f)
        config = configuration_dict[args.configuration]

        if args.model == "scale_label_reader":
            if args.test:
                test_label_reading(args.configuration)
            else:
                train_crnn(configuration=config)
    elif args.command == "view":
        try:
            from django.core.management import execute_from_command_line
        except ImportError as exc:
            raise ImportError(
                "Couldn't import Django. Are you sure it's installed and "
                "available on your PYTHONPATH environment variable? Did you "
                "forget to activate a virtual environment?"
            ) from exc
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'exsclaim.ui.exsclaim_gui.settings')

        if args.bind:
            modify_database_configuration(args.configuration)
        if args.initialize_postgres:
            initialize_database(args.configuration)
            execute_from_command_line(["manage.py", "makemigrations"])
            execute_from_command_line(["manage.py", "migrate"])
        execute_from_command_line(["manage.py", "runserver"])
    elif args.command == "test":
        tests = unittest.defaultTestLoader.discover(parent_directory)
        runner = unittest.TextTestRunner()
        result = runner.run(tests)
