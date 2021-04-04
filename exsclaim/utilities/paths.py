import pathlib

def find_results_dir():
    current_file = pathlib.Path(__file__).resolve(strict=True)
    base_dir = current_file.parent.parent.parent
    results_dirs_file = base_dir / "exsclaim" / "results_dirs"
    with open(results_dirs_file, "r") as f:
        results_dirs = [line.strip() for line in f.readlines()]
    if results_dirs != []:
        results_dir = pathlib.Path(results_dirs[-1])
    else:
        results_dir = base_dir / "extracted"
    return results_dir

def add_results_dir(results_dir):
    """ Add results_dir (a full path) to store extractions in """
    current_file = pathlib.Path(__file__).resolve(strict=True)
    base_dir = current_file.parent.parent.parent
    results_dirs_file = base_dir / "exsclaim" / "results_dirs"
    with open(results_dirs_file, "a") as f:
        f.write(results_dir)