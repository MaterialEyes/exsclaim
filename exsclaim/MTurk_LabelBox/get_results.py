import boto3
import json

## Naming conventions
# LabelBox
LabelBoxMaster = "Master Image"
LabelBoxDependent = "Dependent Image"
LabelBoxInset = "Inset Image"
LabelBoxSubLabel = "Subfigure Label"
LabelBoxScaleLabel = "Scale Bar Label (i.e. 10 nm)"
LabelBoxScaleBar = "Scale Bar Line (actual bar or line)"
# Mechanical Turk
MTurkMaster = "master"
MTurkDependent = "dependent"
MTurkInset = "inset"
MTurkSubLabel = "subfigure_label"
MTurkScaleLabel = "scale_bar_label"
MTurkScaleBar = "scale_bar"

## Delete in actual deployment
MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

# Create your connection to MTurk
mtc = boto3.client('mturk', aws_access_key_id='',
aws_secret_access_key='',
region_name='us-east-1', 
endpoint_url = MTURK_SANDBOX)

## This is the HIT (Human Intelligence Task) ID whose results we are looking for
hit_ids = ["3538U0YQ1FP9FLLG88Z8EGUMCEW3F5"]

def labelbox_json_from_hitid(hit_id):
	""" calls MTurk to results for HIT and converts them to labelbox format """
	output = []
	result = mtc.list_assignments_for_hit(HITId = hit_id)
	for assignment in result['Assignments']:
		labelbox_json = convert_single_image(assignment)
		output.append(labelbox_json)
	return output
	

def convert_single_image(assignment):
	""" converts worker_answer from MTurk to LabelBox format
	
	param image_name: name of the image_name
	param worker_answer: python dictionary from worker json
	
	returns python dictionary in LabelBox JSON format
	"""
	xml_answer = assignment['Answer']
	## strips unused parts of XML response
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
             "External ID": "no name",
             "ID": None,
			 "Label" : single_image_label }
	return data


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
	
def get_labelbox_json(HITLayoutId):
	# Create your connection to MTurk
	hit_list = []
	mtc = boto3.client('mturk', aws_access_key_id='AKIAQ5NBZORHXCUYHFWO',
					   aws_secret_access_key='PU5H4y8/ZHhsl5XxiTyDXTl8wzvMknHsgme5PxXN',
					   region_name='us-east-1', 
					   endpoint_url = MTURK_SANDBOX)
	hits = mtc.list_hits()
	for hit in hits['HITs']:
		if hit['HITLayoutId'] == HITLayoutId:
			hit_list.append(hit['HITId'])
	
	output = []
	for hit in hit_list:
		output += labelbox_json_from_hitid(hit)
	
	return output
		
	
print(get_labelbox_json('3N1982R7HNS4XE89MKMATBZKS9PB5K'))
	











