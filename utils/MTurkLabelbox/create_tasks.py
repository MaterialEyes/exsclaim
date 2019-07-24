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
ap.add_argument("-i", "--image_names", type=str, 
				default="image_urls_to_id_name.txt",
				help="Text file with names of images to send to mturk " + 
				"for labeling")
ap.add_argument("-d", "--deploy", type=str, default="False",
				help="enter true, y, or 1 if you are deploying. false, " + 
				"n, or 0 to test")
ap.add_argument("-l", "--layout_id", type=str, 
				help="enter the layout id available by clicking on the" +
				     "project name in your requester account")
ap.add_argument("-t", "--type_id", type=str, 
				help="enter the type id available by clicking on the" +
				     "project name in your requester account")
args = vars(ap.parse_args())

# constants
api_key = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjand1bzkweHJiMDR" + 
           "rMDgzNmFtdTdhMGVwIiwib3JnYW5pemF0aW9uSWQiOiJjang0dnJoejZjd21sMDg0NDh" +
		   "oODJiMzM3IiwiYXBpS2V5SWQiOiJjang2NDY5emtlMXNqMDgxMXdibWR3NHRwIiwiaWF" +
		   "0IjoxNTYxMTIyNzQzLCJleHAiOjIxOTIyNzQ3NDN9.Wgshf25Ls_eoPO21LaD810OtoH" + 
		   "uxgvwHnsFyHDb4kjw")	
dataset_id = 'cjx65cbxsfgmb0800604zurzo'

# parse command line arguments
access_key = args["access_key"]
secret_key = args["secret_key"]
file_name = args["image_names"]
layout_id = args["layout_id"]
type_id = args["type_id"]
testing = args["deploy"]


# Names of desired images
good_images = set()
directory_name = "150419_3"
directory = os.fsencode(directory_name)
for file in os.listdir(directory):
	good_images.add(os.fsdecode(file))


# generate list of image_urls (hosted on an AWS s3 bucket)
def get_naming_dictionary():
	f = open(file_name, "r")
	json_string = f.read().replace("'","\"")
	json_string = json_string.replace("(", "[")
	json_string = json_string.replace(")", "]")
	f.close()
	return json.loads(json_string)
naming_dictionary = get_naming_dictionary()
image_urls = []

for key in naming_dictionary:
	if naming_dictionary[key][1] in good_images:
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
created = []


# Create an HIT for each image url
completed = 0
for image in image_urls:
    if completed > 100:
        break
    if check_existing(image):
        try:
            response = mtc.create_hit_with_hit_type(
            HITLayoutId    = layout_id,
            HITLayoutParameters = [ {'Name': 'image_url', 'Value': image } ],
            HITTypeId      = type_id,
            LifetimeInSeconds = 2592000
            )
            print(response["HIT"]["HITId"] + " " + image)
            created.append(image)
            completed += 1
        except:
            time.sleep(65)
            response = mtc.create_hit_with_hit_type(
            HITLayoutId    = layout_id,
            HITLayoutParameters = [ {'Name': 'image_url', 'Value': image } ],
            HITTypeId      = type_id,
            LifetimeInSeconds = 2592000
            )
            print(response["HIT"]["HITId"] + " " + image)
            created.append(image)
            completed += 1
			
with open("created_hits_v2.txt", "w") as f:
    f.write(str(created))