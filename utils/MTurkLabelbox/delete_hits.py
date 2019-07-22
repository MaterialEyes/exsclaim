import boto3
import json
import argparse
from graphqlclient import GraphQLClient
import time
import datetime


# for command line usage
ap = argparse.ArgumentParser()


ap.add_argument("-d", "--deploy", type=str, default="False",
				help="enter true, y, or 1 if you are deploying. false, " + 
				"n, or 0 to test")

args = vars(ap.parse_args())


# parse command line arguments
access_key = "AKIARMNOR4HV4ZFO6PN6"
secret_key = "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"

testing = args["deploy"]


# determines endpoint_url
if testing.lower() in ["true", "y", "yes", "1", "yeah", "t"]:
	endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
elif testing.lower() in ["false", "n", "no", "0", "nope", "f"]:
	endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
else:
	raise argparse.ArgumentTypeError("please provide a boolean " + 
	                                 "value for '--testing'")

## Naming conventions
# LabelBox
LabelBoxMaster = "Master Image"
LabelBoxDependent = "Dependent Image"
LabelBoxInset = "Inset Image"
LabelBoxSubLabel = "Subfigure Label"
LabelBoxScaleLabel = "Scale Bar Label"
LabelBoxScaleBar = "Scale Bar Line"
# Mechanical Turk
MTurkMaster = "master"
MTurkDependent = "dependent"
MTurkInset = "inset"
MTurkSubLabel = "subfigure_label"
MTurkScaleLabel = "scale_bar_label"
MTurkScaleBar = "scale_bar"

# Constants
api_key = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjand1bzkweHJiMDR" + 
		  "rMDgzNmFtdTdhMGVwIiwib3JnYW5pemF0aW9uSWQiOiJjang0dnJoejZjd21sMDg0NDh" +
		  "oODJiMzM3IiwiYXBpS2V5SWQiOiJjang2NDY5emtlMXNqMDgxMXdibWR3NHRwIiwiaWF" +
		  "0IjoxNTYxMTIyNzQzLCJleHAiOjIxOTIyNzQ3NDN9.Wgshf25Ls_eoPO21LaD810OtoH" + 
		  "uxgvwHnsFyHDb4kjw")		  
project_id = 'cjx64zn93fdwl0890uxh0agvk'
dataset_id = 'cjx65cbxsfgmb0800604zurzo'
mtc = boto3.client('mturk', aws_access_key_id=access_key,
					   aws_secret_access_key=secret_key,
					   region_name='us-east-1', 
					   endpoint_url = endpoint_url)
					   

with open("hit_published.txt", "r") as f:
    hit_list = f.read().split("\n")
	
    for hit in hit_list:
        hit = hit.split()
        hit = hit[0]
        try: 
            response = mtc.update_expiration_for_hit(HITId=hit,ExpireAt=datetime.datetime(2015, 1, 1))
        except: 
            time.sleep(5)
            response = mtc.update_expiration_for_hit(HITId=hit,ExpireAt=datetime.datetime(2015, 1, 1))