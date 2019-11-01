import boto3
import argparse
import json
import time
import os
from graphqlclient import GraphQLClient


# for command line usage
ap = argparse.ArgumentParser()
ap.add_argument("-k", "--access_key", type=str,
                help="AWS Access Key ID")
ap.add_argument("-s", "--secret_key", type=str,
                help="AWS Secret Access Key")
ap.add_argument("-d", "--deploy", type=str, default="True",
				help="enter true, y, or 1 if you are deploying. false, " + 
				"n, or 0 to test")
ap.add_argument("-l", "--layout_id", type=str, 
				help="enter the layout id available by clicking on the" +
				     "project name in your requester account")
ap.add_argument("-t", "--type_id", type=str, 
				help="enter the type id available by clicking on the" +
				     "project name in your requester account")
ap.add_argument("-n", "--number", type=int, default=100, help="number of hits to make")
ap.add_argument("--name", type=str, help="name of dataset to use")
ap.add_argument("--version", "-v", type=int, default=1, help="number of releases for this dataset")
args = vars(ap.parse_args())

# parse command line arguments
access_key = args["access_key"]
secret_key = args["secret_key"]
layout_id = args["layout_id"]
type_id = args["type_id"]
testing = args["deploy"]
limit = args["number"]
name = args["name"]
version = args["version"]

# constants
api_key = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjand1bzkweHJiMDR" + 
           "rMDgzNmFtdTdhMGVwIiwib3JnYW5pemF0aW9uSWQiOiJjang0dnJoejZjd21sMDg0NDh" +
		   "oODJiMzM3IiwiYXBpS2V5SWQiOiJjang2NDY5emtlMXNqMDgxMXdibWR3NHRwIiwiaWF" +
		   "0IjoxNTYxMTIyNzQzLCJleHAiOjIxOTIyNzQ3NDN9.Wgshf25Ls_eoPO21LaD810OtoH" + 
		   "uxgvwHnsFyHDb4kjw")	
with open("dataset_name_to_id.json", "r") as f:
    dataset_name_to_id = json.load(f)   
dataset_id = dataset_name_to_id[name]

naming_path = os.path.join("naming_jsons", name + ".json")

with open(naming_path, "r") as f:
    naming_dictionary = json.load(f)

image_urls = []

for key in naming_dictionary:
	image_urls.append(key)

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
	
	
# check a label hasn't been made already
def check_existing(image):
    client = GraphQLClient('https://api.labelbox.com/graphql')
    client.inject_token('Bearer ' + api_key)
    if get_existing_labels(client, dataset_id, naming_dictionary[image][0]) == []:
        return True
    else:
        return False
	
# determines endpoint_url
if testing.lower() in ["true", "y", "yes", "1", "yeah", "t"]:
	endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
elif testing.lower() in ["false", "n", "no", "0", "nope", "f"]:
	endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
else:
	raise argparse.ArgumentTypeError("please provide a boolean " + 
	                                 "value for '--testing'")
	

# Create your connection to MTurk
mtc = boto3.client('mturk', aws_access_key_id=access_key,
aws_secret_access_key=secret_key,
region_name='us-east-1', 
endpoint_url = endpoint_url)



# Create an HIT for each image url
completed = 0
hitid_to_url = {}
for image in image_urls:
    if completed >= limit:
        break
    if check_existing(image):
        try:
            response = mtc.create_hit_with_hit_type(
            HITLayoutId    = layout_id,
            HITLayoutParameters = [ {'Name': 'image_url', 'Value': image } ],
            HITTypeId      = type_id,
            LifetimeInSeconds = 2592000
            )
            hitid_to_url[response["HIT"]["HITId"]] = image
            print(response["HIT"]["HITId"] + " " + image)
            completed += 1
        except:
            time.sleep(65)
            response = mtc.create_hit_with_hit_type(
            HITLayoutId    = layout_id,
            HITLayoutParameters = [ {'Name': 'image_url', 'Value': image } ],
            HITTypeId      = type_id,
            LifetimeInSeconds = 2592000
            )
            hitid_to_url[response["HIT"]["HITId"]] = image
            print(response["HIT"]["HITId"] + " " + image)
            completed += 1
			
os.makedirs("hitid_to_url", exist_ok = True)           
output_path = os.path.join("hitid_to_url", name + "_v{}".format(version) + ".json")
with open(output_path, "w") as f:
    json.dump(hitid_to_url, f)
