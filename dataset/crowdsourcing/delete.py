import os
import boto3
import json
import argparse
from graphqlclient import GraphQLClient
import time
import datetime


# for command line usage
ap = argparse.ArgumentParser()


ap.add_argument("-H", "--hit_file", type=str, 
                help="enter hit id dest file")
ap.add_argument("-k", "--access_key", type=str,
                help="AWS Access Key ID")
ap.add_argument("-s", "--secret_key", type=str,
                help="AWS Secret Access Key")
ap.add_argument("-n", "--name", type=str, 
                help="name of dataset to get results for")
ap.add_argument("-v", "--version", type=int, default = 1, help="dataset release version for which you want results")

args = vars(ap.parse_args())


# parse command line arguments
access_key = args["access_key"]
secret_key = args["secret_key"]
name = args["name"]
version = args["version"]
		   
# Constants
endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
mtc = boto3.client('mturk', aws_access_key_id=access_key,
                       aws_secret_access_key=secret_key,
                       region_name='us-east-1', 
                       endpoint_url = endpoint_url)

hit_file = os.path.join("hitid_to_url", name + "_v{}".format(version) + ".json")
with open(hit_file, "r") as f:
    hit_dict = json.load(f)	
    for hit in hit_dict:
        print(hit)
        try: 
            response = mtc.update_expiration_for_hit(HITId=hit,ExpireAt=datetime.datetime(2015, 1, 1))
        except: 
            time.sleep(5)
            response = mtc.update_expiration_for_hit(HITId=hit,ExpireAt=datetime.datetime(2015, 1, 1))