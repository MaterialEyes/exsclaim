import json
import argparse
import time
import os

from graphqlclient import GraphQLClient
import boto3


# for command line usage
ap = argparse.ArgumentParser()
ap.add_argument("-k", "--access_key", type=str,
                help="AWS Access Key ID")
ap.add_argument("-s", "--secret_key", type=str,
                help="AWS Secret Access Key")
ap.add_argument("-n", "--name", type=str, 
                help="name of dataset")
ap.add_argument("-v", "--version", type=int, default=1, help="version number")    
				
args = vars(ap.parse_args())


# parse command line arguments
access_key = args["access_key"]
secret_key = args["secret_key"]
dataset_name = args["name"]
version = args["version"]



# Constants
api_key = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjand1bzkweHJiMDR" + 
		  "rMDgzNmFtdTdhMGVwIiwib3JnYW5pemF0aW9uSWQiOiJjang0dnJoejZjd21sMDg0NDh" +
		  "oODJiMzM3IiwiYXBpS2V5SWQiOiJjang2NDY5emtlMXNqMDgxMXdibWR3NHRwIiwiaWF" +
		  "0IjoxNTYxMTIyNzQzLCJleHAiOjIxOTIyNzQ3NDN9.Wgshf25Ls_eoPO21LaD810OtoH" + 
		  "uxgvwHnsFyHDb4kjw")		  
project_id = 'cjx64zn93fdwl0890uxh0agvk'
with open("dataset_name_to_id.json", "r") as f:
    dataset_name_to_id = json.load(f)
dataset_id = dataset_name_to_id[dataset_name]
endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
mtc = boto3.client('mturk', aws_access_key_id=access_key,
					   aws_secret_access_key=secret_key,
					   region_name='us-east-1', 
					   endpoint_url = endpoint_url)
                       
naming_file = os.path.join("naming_jsons", dataset_name + ".json")                       
with open(naming_file, "r") as f:
	naming_dict = json.load(f)


hit_file = os.path.join("hitid_to_url", dataset_name + "_v{}".format(version) + ".json")
with open(hit_file, "r") as f:
	hit_list = eval(f.read())
	url_to_assignment = {}
	for hit in hit_list:
		if hit != "":
			url = hit_list[hit]
			response = mtc.list_assignments_for_hit(HITId = hit)
			if response['Assignments'] != []:
				assignment_id = response['Assignments'][0]['AssignmentId']
				url_to_assignment[url] = assignment_id

internal_id_to_assignment_id = {}
for url in naming_dict:
	if url in url_to_assignment:
		internal_id, name = naming_dict[url]
		internal_id_to_assignment_id[internal_id] = url_to_assignment[url]
	

results_file = os.path.join("results", dataset_name + "_v{}".format(version) + ".json")
with open(results_file, "r") as f:
	output_json = json.load(f)
	

def reject_assignment_list(assignment_list):
	for assignment_id in assignment_list:
		try:
			mtc.reject_assignment(AssignmentId = assignment_id, RequesterFeedback= "The HIT requires all objects" +
												"of interest to be identified with accurate bounding boxes. Further " +
                                                "we asked for text to be transcribed exactly as it appears ('(a)' should be" +
                                                "written as '(a)' and '10 nm' as '10 nm' not as 'a' and '10nm'." + 
                                                " We were lenient this time and only rejected if there were mulitple types of" +
                                                " errors that would take us a long time to correct.")
		except:
			print("{} already reviewed".format(assignment_id))
	print("Rejected {} assignments".format(len(assignment_list)))
		
def accept_assignment_list(assignment_list):	
    for assignment_id in assignment_list:
        try:
            # mtc.approve_assignment(AssignmentId = assignment_id, RequesterFeedback="Thank you for you interest in our project!" +
            #                                        " We will soon release more HITs with instructions emphasizing our desires.")	
            mtc.approve_assignment(AssignmentId = assignment_id, RequesterFeedback="Thank you for your high-quality work and general interest in our project!")
        except:
            print("{} already reviewed".format(assignment_id))
    print("Approved {} assignments".format(len(assignment_list)))
	
	
def get_existing_labels(client, dataset_id, datarow_id):
	""" returns dataRow externalId to id dict from index:index+100 """
	res_str = client.execute("""
	query getDatasetInformation($dataset_id: ID!, $datarow_id: ID!){
		dataset(where: {id: $dataset_id}){
			dataRows(where: {id: $datarow_id}) {
				labels{
					id
				}
			}
		}
	}
    """, {'dataset_id': dataset_id, 'datarow_id': datarow_id} )

	res = json.loads(res_str)
	return res["data"]["dataset"]["dataRows"][0]["labels"]
	
	
def get_worker_labelIds(labelbox_json):
	client = GraphQLClient('https://api.labelbox.com/graphql')
	client.inject_token('Bearer ' + api_key)
	worker_to_unapproved = {}
	worker_to_approved = {}
	for label in labelbox_json:
		if get_existing_labels(client, dataset_id, label["ID"]) == []:
			internal_ids = worker_to_unapproved.get(label["Created By"], [])
			internal_ids.append(internal_id_to_assignment_id[label["ID"]])
			worker_to_unapproved[label["Created By"]] = internal_ids
		else:
			internal_ids = worker_to_approved.get(label["Created By"], [])
			internal_ids.append(internal_id_to_assignment_id[label["ID"]])
			worker_to_approved[label["Created By"]] = internal_ids			
	return worker_to_unapproved, worker_to_approved

approvals = []
rejections = []
worker_to_unapproved, worker_to_approved = get_worker_labelIds(output_json)
workers = {i for i in worker_to_approved}
workers.update({j for j in worker_to_unapproved})
for i in workers:
	print("Worker {}, {} approved, {} rejected".format(i, len(worker_to_approved.get(i, [])), len(worker_to_unapproved.get(i, []))))	
	approvals += worker_to_approved.get(i, [])
	rejections += worker_to_unapproved.get(i, [])
	
reject_assignment_list(rejections)
accept_assignment_list(approvals)
