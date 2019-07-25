import boto3
import json
import argparse
from graphqlclient import GraphQLClient
import time


# for command line usage
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--worker_id", type=str,
                help="worker id to delete")

args = vars(ap.parse_args())

# parse command line arguments
worker = args["worker_id"]


# Constants
api_key = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjand1bzkweHJiMDR" + 
		  "rMDgzNmFtdTdhMGVwIiwib3JnYW5pemF0aW9uSWQiOiJjang0dnJoejZjd21sMDg0NDh" +
		  "oODJiMzM3IiwiYXBpS2V5SWQiOiJjang2NDY5emtlMXNqMDgxMXdibWR3NHRwIiwiaWF" +
		  "0IjoxNTYxMTIyNzQzLCJleHAiOjIxOTIyNzQ3NDN9.Wgshf25Ls_eoPO21LaD810OtoH" + 
		  "uxgvwHnsFyHDb4kjw")		 
client = GraphQLClient('https://api.labelbox.com/graphql')
client.inject_token('Bearer ' + api_key)		  
project_id = 'cjx64zn93fdwl0890uxh0agvk'
dataset_id = 'cjx65cbxsfgmb0800604zurzo'

					   

workers_to_data_id = {}
with open("hit_published_output.json", "r") as f:
	labelbox_json = json.load(f)

for entry in labelbox_json:
	worker_results = workers_to_data_id.get(entry["Created By"], [])
	worker_results.append(entry["ID"])
	workers_to_data_id[entry["Created By"]] = worker_results
	



	
def delete_labels(client, dataset_id, datarow_id):
	""" returns dataRow externalId to id dict from index:index+100 """
	res_str = client.execute("""
	query getLabelInformation($dataset_id: ID!, $datarow_id: ID!){
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
	try:
		label_id =  res["data"]["dataset"]["dataRows"][0]["labels"][0]["id"]
	except:
		label_id = ""
	res_str = client.execute("""
	mutation DeleteLabelsFromAPI($label_ids: [ID!]!) {
		deleteLabels(labelIds: $label_ids
		){
			id
			deleted
		}
	}
    """, {'label_ids': [label_id]} )

for image in workers_to_data_id[worker]:
		delete_labels(client, dataset_id, image))

print("Deleted {} images from Labelbox".format(len(workers_to_data_id[worker])))
