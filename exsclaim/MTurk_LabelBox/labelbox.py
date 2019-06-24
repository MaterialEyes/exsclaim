import json
from graphqlclient import GraphQLClient
api_key = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjand1bzkweHJiMDR" + 
		  "rMDgzNmFtdTdhMGVwIiwib3JnYW5pemF0aW9uSWQiOiJjang0dnJoejZjd21sMDg0NDh" +
		  "oODJiMzM3IiwiYXBpS2V5SWQiOiJjang2NDY5emtlMXNqMDgxMXdibWR3NHRwIiwiaWF" +
		  "0IjoxNTYxMTIyNzQzLCJleHAiOjIxOTIyNzQ3NDN9.Wgshf25Ls_eoPO21LaD810OtoH" + 
		  "uxgvwHnsFyHDb4kjw")
		  
project_id = 'cjx64zn93fdwl0890uxh0agvk'
dataset_id = 'cjx65cbxsfgmb0800604zurzo'
client = GraphQLClient('https://api.labelbox.com/graphql')
client.inject_token('Bearer ' + api_key)

	
def get_data_row_ids(index):
	""" returns dataRow externalId to id dict from index:index+100 """
	res_str = client.execute("""
	query getDatasetInformation($dataset_id: ID!){{
		dataset(where: {{id: $dataset_id}}){{
			name
			id
			dataRows(skip: {}) {{
				id
				externalId
				}}
		}}
	}}

    """.format(str(index)), {'dataset_id': dataset_id} )

	res = json.loads(res_str)
	return res['data']['dataset']['dataRows']

def get_externalId_dict():
	externalId_to_id = {}	
	for i in range(0,38):
		labelbpx_dictlist = get_data_row_ids(i*100)
		for dict in labelbpx_dictlist:
			externalId_to_id[dict['externalId']] = dict['id']
	return externalId_to_id

print(get_externalId_dict())
	
