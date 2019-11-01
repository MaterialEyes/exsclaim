import boto3
import json
import argparse
from graphqlclient import GraphQLClient
import time
import collections


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

args = vars(ap.parse_args())

# parse command line arguments
access_key = args["access_key"]
secret_key = args["secret_key"]
ids = args["assignments"]
id_type = args["type"]


# Constants
endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
mtc = boto3.client('mturk', aws_access_key_id=access_key,
					   aws_secret_access_key=secret_key,
					   region_name='us-east-1', 
					   endpoint_url = endpoint_url)

def review(id_list, id_type, mtc):
    assignments = []
    workers = []
    reviewable = 0
    assignable = 0
    unassignable = 0
    reviewing = 0
    disposed = 0

    # if the id list of Worker Ids
    if id_type.lower() in ["worker", "workers", "w"]:
        with open("worker_data.json", "r") as f:
            worker_dict = json.load(f)
        for worker in id_list:
            assignments += worker_dict[worker]

    # if the id list is of HITs
    elif id_type.lower() in ["hit", "h", "hits"]:
        for hid in id_list:
            res = mtc.get_hit(HITId = hid)
            # print("Status: ",res["HIT"]["HITStatus"])
            # print("Creation",res["HIT"]["CreationTime"])
            # print("Expiration: ",res["HIT"]["Expiration"])
            # print("Review Status: ",res["HIT"]["HITReviewStatus"])
            # print("Assignments Pending: ",res["HIT"]["NumberOfAssignmentsPending"])
            # print("Assignments Available: ",res["HIT"]["NumberOfAssignmentsAvailable"])
            # print("Assignments Completed: ",res["HIT"]["NumberOfAssignmentsCompleted"])
            afh = mtc.list_assignments_for_hit(HITId = hid)
            try:
                a = afh["Assignments"][0]
                # print("Worker: ",a['WorkerId'])
                workers.append(a['WorkerId'])
            except:
                dummy = ""
                # print("Worker: ","Unassigned")

            # print("\n")

            if res["HIT"]["HITStatus"] == "Unassignable":
                unassignable += 1
            elif res["HIT"]["HITStatus"] == "Assignable":
                assignable += 1
            elif res["HIT"]["HITStatus"] == "Reviewable":
                reviewable += 1
            elif res["HIT"]["HITStatus"] == "Reviewing":
                reviewing += 1
            elif res["HIT"]["HITStatus"] == "Disposed":
                disposed += 1
            else:
                print(res["HIT"]["HITStatus"]+" status not counted!")

        print("Unassignable: ",unassignable)
        print("Assignable:   ",assignable)
        print("Reviewable:   ",reviewable)
        print("Reviewing:    ",reviewing)
        print("Disposed:     ",disposed)
        print("-------------------")
        print("Total:        ",unassignable+assignable+reviewable+reviewing+disposed,"\n")

        print(collections.Counter(workers))

    elif id_type.lower() in ["assignment", "a", "assignments"]:
        assignments = id_list
    else:
        print("invalid id_type. Must be hit, worker, or assignment")
    return assignable  

# with open(ids, "r",encoding='utf-8-sig') as f:
#     id_list = [a.strip("\n") for a in f.read().split(",")]
    
if __name__ == '__main__':
    with open(ids, "r",encoding='utf-8-sig') as f:
        id_list = [a.strip("\n") for a in f.read().split(",")]
    print(review(id_list, id_type, mtc))
    
