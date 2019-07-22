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
ap.add_argument("-f", "--format", type=str, default="0",
				help="enter how you would like to get the results:\n " +
					 "0 - output to terminal\n1 - write to output.json\n" +
					 "2 - upload to labelbox\n3 - output images with" +
					 "bounding boxes drawn to output\nYou may enter multiple")
ap.add_argument("-d", "--deploy", type=str, default="False",
				help="enter true, y, or 1 if you are deploying. false, " + 
				"n, or 0 to test")
ap.add_argument("-H", "--hit_file", type=str, 
                help="enter hit id dest file")
args = vars(ap.parse_args())


# parse command line arguments
access_key = args["access_key"]
secret_key = args["secret_key"]
format = args["format"]
testing = args["deploy"]
hit_file = args["hit_file"]

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
					   
def get_naming_dictionary():
	f = open("image_urls_to_id_name.txt", "r")
	json_string = f.read().replace("'","\"")
	json_string = json_string.replace("(", "[")
	json_string = json_string.replace(")", "]")
	f.close()
	return json.loads(json_string)
	
naming_dict = get_naming_dictionary()
new_worker_data = {}


<<<<<<< HEAD

# Create your connection to MTurk
mtc = boto3.client('mturk', aws_access_key_id='',
aws_secret_access_key='',
region_name='us-east-1', 
endpoint_url = endpoint_url)


def labelbox_json_from_hitid(hit_id):
=======
def labelbox_json_from_hitid(hit_id, url):
>>>>>>> b951d98... Finished code to upload mturk labels to labelbox
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
<<<<<<< HEAD
<<<<<<< HEAD
	hit_list = []
	mtc = boto3.client('mturk', aws_access_key_id=access_key,
					   aws_secret_access_key=secret_key,
					   region_name='us-east-1', 
					   endpoint_url = MTURK_SANDBOX)
	hits = mtc.list_hits()
	for hit in hits['HITs']:
		if hit['HITLayoutId'] == HITLayoutId:
			hit_list.append(hit['HITId'])
	
=======
	f = open("hitids.txt", "r")
=======
	f = open(hit_file, "r")
>>>>>>> 249e684... Updated mturk files
	hit_list = f.read().split("\n")
>>>>>>> b951d98... Finished code to upload mturk labels to labelbox
	output = []
	error = []
	for line in hit_list:
		if line != "":
			hit, url = line.split()
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
		

# uses helper functions to retrieve labelbox_json for completed HITs	
labelbox_json, errors = get_labelbox_json()

# displays results in desired formats
if "0" in format:
	print(json.dumps(labelbox_json, sort_keys="True", indent=2))
	print("There were errors...", errors)
if "1" in format:
	# write results to output file
	output_file = hit_file.split(".")[0] + "_output.json"
	with open(output_file, "w") as g:
		json.dump(labelbox_json, g)
	
	# write dictionary mapping workers to hit's to a json 
	with open("worker_data.json", "r") as h:
		old_data = json.load(h)
	
	for worker in new_worker_data:
		if worker in old_data:
			old_data[worker].add(new_worker_data[worker])
		else:
			old_data[worker] = new_worker_data[worker]
	
	with open("worker_data.json", "w") as h:
		json.dump(old_data, h)
	
if "2" in format:
	print("sending to https://app.labelbox.com/projects/cjx64zn93fdwl0890uxh0agvk/labels/activity")
<<<<<<< HEAD
	upload_labels(labelbox_json)
=======
	print("Some images already have labels in Labelbox and were not uploaded: ", upload_labels(labelbox_json))
	print("These HITs had errors in the formatting of results: ", errors)
>>>>>>> b092c71... Added tracking what hits each worker has done
if "3" in format:
	print("format '3' not implemented")
	
