import boto3
import json
import argparse
from graphqlclient import GraphQLClient
import time


# for command line usage
ap = argparse.ArgumentParser()
ap.add_argument("-k", "--access_key", type=str,
                help="AWS Access Key ID")
ap.add_argument("-s", "--secret_key", type=str,
                help="AWS Secret Access Key")
ap.add_argument("-A", "--accept", type=str, 
                help="path to csv of worker ids to accept")
ap.add_argument("-R", "--reject", type=str, 
                help="path to csv of worker ids to reject")
ap.add_argument("-B", "--ban", type=str, 
                help="path to csv of worker ids to ban")

args = vars(ap.parse_args())


# parse command line arguments
access_key = args["access_key"]
secret_key = args["secret_key"]
accept = args["accept"]
reject = args["reject"]
ban = args["ban"]

# Constants
endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
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

					   
def get_naming_dictionary():
	f = open("image_urls_to_id_name.txt", "r")
	json_string = f.read().replace("'","\"")
	json_string = json_string.replace("(", "[")
	json_string = json_string.replace(")", "]")
	f.close()
	return json.loads(json_string)
	
naming_dict = get_naming_dictionary()

## Get list of rejected and accepted HITs
with open("worker_data.json", "r") as f:
	worker_data = json.load(f)
	
with open(accept, "r") as f:
	full_text = f.read()
	accept_worker_list = full_text.split(",")
	## get the worker's hits
	accept_assignemnts = []
	for worker in accept_worker_list:
		accept_assignemnts += worker_data.get(worker, [])

with open(reject, "r") as f:
	full_text = f.read()
	reject_worker_list = full_text.split(",")
	## get the worker's hits
	reject_assignments = []
	for worker in reject_worker_list:
		reject_assignments += worker_data.get(worker, [])

## Get list of banned workers
with open(ban, "r") as f:
	full_text = f.read()
	banned = full_text.split(",")


def reject_assignment_list(assignment_list):
	for assignment_id in assignment_list:
		mtc.reject_assignment(assignment_id, "The HIT requires all objects" +
											 "of interest to be identified")
		
def accept_assignment_list(assignment_list):
	for assignment_id in assignment_list:
		mtc.reject_assignment(assignment_id, "The HIT requires all objects" +
											 "of interest to be identified")		
		
def block_workers(worker_list):
	for worker_id in worker_list:
		mtc.create_worker_block(worker_id, "this worker is a bot")
	


reject_assignment_list(reject_assignments)
accept_assignment_list(accept_assignemnts)
block_workers(banned)
	
