import argparse
import json

import ObjectDetector.run as od
import TextDetector.run as td


def parse_command_line_arguments():
    """ parses arguments input throug command line as dictionary """
    ap = argparse.ArgumentParser()
    
    ap.add_argument('-i', '--input-dir', type=str, default='images',
                    help='Path to directory containing input images')










if __name__ == '__main__':
    ## parse command line args
    args = parse_command_line_arguments()
    
    ## Run ObjectDetector on input images
    data = od.load_and_run_model(args['input_dir'])

    print(data)
