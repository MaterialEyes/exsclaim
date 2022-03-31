"""Handles location of output files"""
import pathlib


def initialize_results_dir(results_dir=None):
    """Determine where to save results for a pipeline run

    The output directory will be resolved in this order:
        1. if results_dir is provided, results will be to results_dir/query_name
        2. results will be saved to exsclaim_repo_root/output/query_name

    Note: On Docker, providing a results_dir may not work as intended,
        as the results will be saved in the docker container. You will
        need to set up a volume mapping your desired results directory
        to the output path in the docker container.

    Args:
        results_dir (str): path to desired results directory, default None.
    Returns:
        results_dir (pathlib.Path): Full path to output directory
    Modifies:
        Creates results_dir if it doesn't exist.
    """
    current_file = pathlib.Path(__file__).resolve(strict=True)
    base_dir = current_file.parent.parent.parent

    if results_dir:
        results_dir = pathlib.Path(results_dir).resolve()
    else:
        results_dir = base_dir / "output"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir
