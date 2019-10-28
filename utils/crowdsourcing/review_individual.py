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
ap.add_argument("-a", "--assignments", type=str, 
                help="path to csv of ids")
ap.add_argument("-t", "--type", type=str, 
                help="type of ids, one of: worker, assignment, or HIT")
ap.add_argument("-r", "--review", type=str, 
                help="accept, reject, or ban. Ban only works for worker IDs.")
ap.add_argument("-o", "--override", type=str, default=False,
                help="true to override rejections")
ap.add_argument("-m", "--message", type=str, default="",
                help="message to send to workers with review")
args = vars(ap.parse_args())

# parse command line arguments
access_key = args["access_key"]
secret_key = args["secret_key"]
ids = args["assignments"]
id_type = args["type"]
review_action = args["review"]
override = args["override"]
message = args["message"]

# Constants
endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
mtc = boto3.client('mturk', aws_access_key_id=access_key,
					   aws_secret_access_key=secret_key,
					   region_name='us-east-1', 
					   endpoint_url = endpoint_url)

def review(id_list, id_type, review_action, message, mtc, override=False):
    assignments = []
    # if the id list of Worker Ids
    if id_type.lower() in ["worker", "workers", "w"]:
        with open("worker_data.json", "r") as f:
            worker_dict = json.load(f)
        for worker in id_list:
            assignments += worker_dict[worker]

    # if the id list is of HITs
    elif id_type.lower() in ["hit", "h", "hits"]:
        for hid in id_list:
            res = mtc.list_assignments_for_hit(HITId = hid)
            for assignment in res["Assignments"]:
                assignments.append(assignment["AssignmentId"])
    elif id_type.lower() in ["assignment", "a", "assignments"]:
        assignments = id_list
    else:
        print("invalid id_type. Must be hit, worker, or assignment")
    
    # review all assignments
    if review_action.lower() in ["accept", "a", "approve"]:
        accept_assignment_list(assignments, message, override)
    elif review_action.lower() in ["reject", "r", "deny"]:
        reject_assignment_list(assignments, message)
    else:
        print("invalid review. Must be accept or reject or ban")


def reject_assignment_list(assignment_list, message):
	for assignment_id in assignment_list:
		mtc.reject_assignment(AssignmentId = assignment_id, RequesterFeedback= message)
	print("Rejected {} assignments".format(len(assignment_list)))
		
def accept_assignment_list(assignment_list, message, override=False):
	for assignment_id in assignment_list:
		mtc.approve_assignment(AssignmentId = assignment_id, RequesterFeedback=message, OverrideRejection=override)	
	print("Approved {} assignments".format(len(assignment_list)))	
		
def block_workers(worker_list, message):
	for worker_id in worker_list:
		mtc.create_worker_block(WorkerId = worker_id, Reason=message)
	print("Banned {} workers".format(len(worker_list))) 

with open(ids, "r",encoding='utf-8-sig') as f:
    id_list = [a.strip("\n") for a in f.read().split(",")]
    
if review_action.lower() in ["ban", "block", "b"]:
    block_workers(id_list, message)
else:
    review(id_list, id_type, review_action, message, mtc, override)
    
