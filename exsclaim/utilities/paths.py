import pathlib
from sys import path
import os

def initialize_results_dir(results_dir=None):
    """ Determine where to save results for a pipeline run

    The output directory will be resolved in this order:
        1. if results_dir is provided, results will be saved there
        2. if results_dir has previously been provided, results will be
            saved in the most recently added results directory
        3. results will be saved in /path/to/<cwd>/extracted

    Args:
        results_dir (str): path to desired results directory, default None.
    Returns:
        results_dir (pathlib.Path): Full path to output directory
    Modifies:
        Creates results_dir if it doesn't exist. Adds results_dir
        to results_dirs file
    """

    # find all previous results directories
    current_file = pathlib.Path(__file__).resolve(strict=True)
    base_dir = current_file.parent.parent.parent
    results_dirs_file = base_dir / "exsclaim" / "results_dirs"
    if os.path.isfile(results_dirs_file):
        with open(results_dirs_file, "r") as f:
            results_dirs = [line.strip() for line in f.readlines()]
    else:
        results_dirs = []
    # run through results_dir resolution order
    if results_dir:
        results_dir = pathlib.Path(results_dir).resolve()
    elif results_dirs:
        results_dir = pathlib.Path(results_dirs[-1])
    else:
        results_dir = pathlib.Path().cwd() / "extracted"
    results_dir.mkdir(parents=True, exist_ok=True)
    # add results_dir to results_dirs
    if str(results_dir) not in results_dirs:
        add_results_dir(results_dir)
    return results_dir

def add_results_dir(results_dir):
    """ Add results_dir (a full path) to store extractions in """
    current_file = pathlib.Path(__file__).resolve(strict=True)
    base_dir = current_file.parent.parent.parent
    results_dirs_file = base_dir / "exsclaim" / "results_dirs"
    with open(results_dirs_file, "a") as f:
        f.write(str(results_dir) + "\n")