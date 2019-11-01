import json
import argparse
import os
import time

from graphqlclient import GraphQLClient
import boto3



# for command line usage
ap = argparse.ArgumentParser()
ap.add_argument("-k", "--access_key", type=str,
                help="AWS Access Key ID")
ap.add_argument("-s", "--secret_key", type=str,
                help="AWS Secret Access Key")
ap.add_argument("-f", "--format", type=str, default="20",
				help="enter how you would like to get the results:\n " +
					 "0 - output to terminal\n1 - write to output.json\n" +
					 "2 - upload to labelbox\n3 - output images with" +
					 "bounding boxes drawn to output\nYou may enter multiple")
ap.add_argument("-d", "--deploy", type=str, default="True",
				help="enter true, y, or 1 if you are deploying. false, " + 
				"n, or 0 to test")
ap.add_argument("-n", "--name", type=str, 
                help="name of dataset to get results for")
ap.add_argument("-v", "--version", type=int, default = 1, help="dataset release version for which you want results")
args = vars(ap.parse_args())


# parse command line arguments
access_key = args["access_key"]
secret_key = args["secret_key"]
format = args["format"]
testing = args["deploy"]
name = args["name"]
version = args["version"]

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
with open("dataset_name_to_id.json", "r") as f:
    dataset_name_to_id = json.load(f)
dataset_id = dataset_name_to_id[name]

mtc = boto3.client('mturk', aws_access_key_id=access_key,
					   aws_secret_access_key=secret_key,
					   region_name='us-east-1', 
					   endpoint_url = endpoint_url)
					   

naming_file = os.path.join("naming_jsons", name + ".json")
with open(naming_file, "r") as f:
    naming_dict = json.load(f)
new_worker_data = {}


def labelbox_json_from_hitid(hit_id, url):
    """ calls MTurk to results for HIT and converts them to labelbox format """
    output = []
    errors = []
    try: 
        result = mtc.list_assignments_for_hit(HITId = hit_id)
    except: 
        time.sleep(65)
        result = mtc.list_assignments_for_hit(HITId = hit_id)
	
    for assignment in result['Assignments']:
        ## keep track of workers
        worker_hits = new_worker_data.get(assignment["WorkerId"], [])
        worker_hits.append(assignment["AssignmentId"])
        new_worker_data[assignment["WorkerId"]] = worker_hits
        try:
            labelbox_json = convert_single_image(assignment, url)
            output.append(labelbox_json)
        except:
            errors.append((url, assignment))
    return output, errors
	

def convert_single_image(assignment, url):
	""" converts worker_answer from MTurk to LabelBox format
	
	param image_name: name of the image_name
	param worker_answer: python dictionary from worker json
	
	returns python dictionary in LabelBox JSON format
	"""
	xml_answer = assignment['Answer']
		
	## strips unused parts of XML response
	try:
		pre, answer_post = xml_answer.split(">[{", 1)
		answer, post = answer_post.split("}]<", 1)
		json_answer_string = "[{" + answer + "}]"

		## manipulates response as a JSON and converts to LabelBox format
		json_answer = json.loads(json_answer_string)
		single_image_label = convert_single_image_label(json_answer)
						  
		
		## pieces various metadata components with single_image_label
		data = { "Agreement" : None, 
				 "Benchmark Agreement": None,
				 "Benchmark ID": None,
				 "Benchmark Reference ID": None,
				 "Created At": None,
				 "Created By": assignment["WorkerId"],
				 "DataRow ID": None,
				 "Dataset Name": "150419_atomic-resolution",
				 "External ID": naming_dict[url][1],
				 "ID": naming_dict[url][0],
				 "Label" : single_image_label }
		return data
	except:
		raise TypeError


def convert_single_image_label(json_answer):
	""" Converts an MTurk style label json to a labelbox label json """
	single_image_label = { LabelBoxMaster : [], LabelBoxDependent : [],
					   LabelBoxInset : [], LabelBoxSubLabel: [],
					   LabelBoxScaleLabel : [], LabelBoxScaleBar : [] }
					   
	for label in json_answer:
		# convert coordinates
		x1 = label["left"]
		x2 = x1 + label["width"]
		y1 = label["top"]
		y2 = y1 + label["height"]
		coordinates = [ {"x" : x1, "y" : y1}, {"x" : x1, "y" : y2}, 
						{"x" : x2, "y" : y2}, {"x" : x2, "y" : y1} ]
		label_dictionary = {"geometry" :  coordinates}
		# add functionality info
		if label["functionality"] == MTurkMaster:
			label_dictionary["classification"] = label["label"]
			single_image_label[LabelBoxMaster].append(label_dictionary)
		elif label["functionality"] == MTurkDependent:
			label_dictionary["classification"] = label["label"]
			single_image_label[LabelBoxDependent].append(label_dictionary)	
		elif label["functionality"] == MTurkInset:
			label_dictionary["classification"] = label["label"]
			single_image_label[LabelBoxInset].append(label_dictionary)	
		elif label["functionality"] == MTurkSubLabel:
			label_dictionary["text"] = label["label"]
			single_image_label[LabelBoxSubLabel].append(label_dictionary)
		elif label["functionality"] == MTurkScaleLabel:
			label_dictionary["text"] = label["label"]
			single_image_label[LabelBoxScaleLabel].append(label_dictionary)
		elif label["functionality"] == MTurkSubLabel:
			label_dictionary["text"] = label["label"]
			single_image_label[LabelBoxSubLabel].append(label_dictionary)
		elif label["functionality"] == MTurkScaleBar:
			single_image_label[LabelBoxScaleBar].append(label_dictionary)
	return single_image_label
	
def get_labelbox_json():
    # Create your connection to MTurk
    hit_file = os.path.join("hitid_to_url", name + "_v{}".format(version) + ".json")
    with open(hit_file, "r") as f:
        hit_dict = json.load(f)
    output = []
    error = []
    for hit in hit_dict:
        url = hit_dict[hit]
        new_output, new_error = labelbox_json_from_hitid(hit, url)
        output += new_output
        error += new_error
    return output, error
	
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
	
	
def upload_labels(labelbox_json):
	client = GraphQLClient('https://api.labelbox.com/graphql')
	client.inject_token('Bearer ' + api_key)
	pre_existing_labels = []
	for label in labelbox_json:
		if get_existing_labels(client, dataset_id, label["ID"]) == []:
			res_str = client.execute("""
			mutation CreateLabelFromApi($label: String!, $projectId: ID!, $dataRowId: ID!){
			  createLabel(data:{
				label:$label,
				secondsToLabel:0,
				project:{
				  connect:{
					id:$projectId
				  }
				}
				dataRow:{
				  connect:{
					id:$dataRowId
				  }
				}
				type:{
				  connect:{
					name:"Any"
				  }
				}
			  }){
			  id
			  }
			}
			""", {
				'label': json.dumps(label["Label"]),
				'projectId': project_id,
				'dataRowId': label["ID"]
			})
		else: 
			pre_existing_labels.append(label["ID"])
	return pre_existing_labels
		

# uses helper functions to retrieve labelbox_json for completed HITs	
labelbox_json, errors = get_labelbox_json()

# displays results in desired formats
if "0" in format:
    print(json.dumps(labelbox_json, sort_keys="True", indent=2))
    print("There were errors...", errors)
if "1" in format:
    # write results to output file
    os.makedirs("results", exist_ok=True)
    output_file = os.path.join("results", name + "_v{}".format(version) + ".json")
    with open(output_file, "w") as g:
        json.dump(labelbox_json, g)
	
    # write dictionary mapping workers to hit's to a json 
    with open("worker_data.json", "r") as h:
        old_data = json.load(h)
	
    for worker in new_worker_data:
        if worker in old_data:
            old_data[worker].append(new_worker_data[worker])
        else:
            old_data[worker] = new_worker_data[worker]
	
    with open("worker_data.json", "w") as h:
        json.dump(old_data, h)
	
if "2" in format:
    print("sending to https://app.labelbox.com/projects/cjx64zn93fdwl0890uxh0agvk/labels/activity")
    print("Some images already have labels in Labelbox and were not uploaded: ", upload_labels(labelbox_json))
    print("These HITs had errors in the formatting of results: ", errors)
if "3" in format:
    print("format '3' not implemented")

