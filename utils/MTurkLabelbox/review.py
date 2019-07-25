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
ap.add_argument("-H", "--hit_file", type=str, 
                help="enter hit id dest file")
				
args = vars(ap.parse_args())


# parse command line arguments
access_key = args["access_key"]
secret_key = args["secret_key"]
hit_file = args["hit_file"]



# Constants
api_key = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjand1bzkweHJiMDR" + 
		  "rMDgzNmFtdTdhMGVwIiwib3JnYW5pemF0aW9uSWQiOiJjang0dnJoejZjd21sMDg0NDh" +
		  "oODJiMzM3IiwiYXBpS2V5SWQiOiJjang2NDY5emtlMXNqMDgxMXdibWR3NHRwIiwiaWF" +
		  "0IjoxNTYxMTIyNzQzLCJleHAiOjIxOTIyNzQ3NDN9.Wgshf25Ls_eoPO21LaD810OtoH" + 
		  "uxgvwHnsFyHDb4kjw")		  
project_id = 'cjx64zn93fdwl0890uxh0agvk'
dataset_id = 'cjx65cbxsfgmb0800604zurzo'
endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
mtc = boto3.client('mturk', aws_access_key_id=access_key,
					   aws_secret_access_key=secret_key,
					   region_name='us-east-1', 
					   endpoint_url = endpoint_url)
with open("urls_to_names.json", "r") as f:
	naming_dict = json.load(f)


with open(hit_file, "r") as f:
	hit_list = f.read().split("\n")
	url_to_assignment = {}
	for line in hit_list:
		if line != "":
			hit, url = line.split()
			response = mtc.list_assignments_for_hit(HITId = hit)
			if response['Assignments'] != []:
				assignment_id = response['Assignments'][0]['AssignmentId']
				url_to_assignment[url] = assignment_id

internal_id_to_assignment_id = {}
for url in naming_dict:
	if url in url_to_assignment:
		internal_id, name = naming_dict[url]
		internal_id_to_assignment_id[internal_id] = url_to_assignment[url]
	
with open(hit_file.split(".")[0] + "_output.json", "r") as f:
	output_json = json.load(f)
	

def reject_assignment_list(assignment_list):
	for assignment_id in assignment_list:
		try:
			mtc.reject_assignment(AssignmentId = assignment_id, RequesterFeedback= "The HIT requires all objects" +
												"of interest to be identified")
		except:
			print("{} already reviewed".format(assignment_id))
	print("Rejected {} assignments".format(len(assignment_list)))
		
def accept_assignment_list(assignment_list):
	try:	
		for assignment_id in assignment_list:
			mtc.approve_assignment(AssignmentId = assignment_id)	
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
