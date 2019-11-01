## For help with labelbox api, go to https://api.labelbox.com/graphql

import argparse
import json
import time
import os
import argparse

import boto3
from graphqlclient import GraphQLClient

# for command line usage
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", type=str,
                help="Name of dataset to be initialized")
args = vars(ap.parse_args())

name = args["name"]

# constants
api_key = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjand1bzkweHJiMDR" + 
           "rMDgzNmFtdTdhMGVwIiwib3JnYW5pemF0aW9uSWQiOiJjang0dnJoejZjd21sMDg0NDh" +
		   "oODJiMzM3IiwiYXBpS2V5SWQiOiJjang2NDY5emtlMXNqMDgxMXdibWR3NHRwIiwiaWF" +
		   "0IjoxNTYxMTIyNzQzLCJleHAiOjIxOTIyNzQ3NDN9.Wgshf25Ls_eoPO21LaD810OtoH" + 
		   "uxgvwHnsFyHDb4kjw")	
client = GraphQLClient('https://api.labelbox.com/graphql')
client.inject_token('Bearer ' + api_key)


def get_dataset_ids(client):
    """ returns dataset name to dataset id dict """
    res_str = client.execute("""
    query getDatasetInformation{
        datasets{
            name
            id
        }
    }
    """ )
    
    dataset_name_to_id = {}
    res = json.loads(res_str)
    for dataset in res["data"]["datasets"]:
        dataset_name_to_id[dataset["name"].strip()] = dataset["id"]
        
    # save the updated dataset ids
    with open("dataset_name_to_id.json", "w") as f:
        json.dump(dataset_name_to_id, f)
    return dataset_name_to_id

def get_external_ids(client, dataset_name):
    """ returns image_name to labelbox id for datarows in dataset_name """
    ## get the id associated with the dataset name
    with open("dataset_name_to_id.json", "r") as f:
        dataset_name_to_id = json.load(f)
    if dataset_name not in dataset_name_to_id:
        dataset_name_to_id = get_dataset_ids(client)
    dataset_id = dataset_name_to_id[dataset_name]
    
    # get the dataset length
    res_str = client.execute("""
    query getDatasetInformation($dataset_id: ID!){
        dataset(where: {id: $dataset_id}){
            rowCount
        }
    }
    """, {'dataset_id': dataset_id} )
    res = json.loads(res_str)
    length = res["data"]["dataset"]["rowCount"]
    
    # we can only request 100 dataRows at a time so repeat
    externalId_to_id = {}
    for index in range(0, max(1,int(length/100))):
        res_str = client.execute("""
        query getDatasetInformation($dataset_id: ID!, $skip: Int){
            dataset(where: {id: $dataset_id}){
                dataRows(skip: $skip){
                    externalId
                    id
                }
            }
        }
        """, {'dataset_id': dataset_id, 'skip': index*100} )
        res = json.loads(res_str)
        
        # add results to dictionary 
        for datarow in res["data"]["dataset"]["dataRows"]:
            externalId_to_id[datarow["externalId"]] = datarow["id"]
    
    return externalId_to_id

def make_master_dict(client, dataset_name):
    externalId_to_id = get_external_ids(client, dataset_name)
    with open("image_urls_names.txt", "r") as f:
        url_name = f.read()
    url_name_pairs = url_name.split("\n")

    master_dict = {}
    for pair in url_name_pairs:
        if pair != "":
            url, name = pair.split(" ", 1)
            if name in externalId_to_id:
                master_dict[url] = (externalId_to_id[name], name)
    return master_dict
      
def save_master_dict(client, dataset_name):
    master_dict = make_master_dict(client, dataset_name)
    os.makedirs("naming_jsons", exist_ok = True)
    output_file = os.path.join("naming_jsons", dataset_name + ".json")
        
    with open(output_file, "w") as f:
        json.dump(master_dict, f)

save_master_dict(client, name)