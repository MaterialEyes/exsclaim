# Setting up Labeling Tasks with Mechanical Turk and LabelBox

## create_tasks.py

Used to create HITs for a specified project on Mechanical Turk.
Project must first be created as a requester at https://requester.mturk.com/create/projects/new
Select "Other", at the very bottom left. 
Select sign in to create project, sign in. 
Enter your desired properties. These determine type_id
In the Design Layout tab, delete the text that is there and paste the text of working_layout.xml there. This determines layout_id.
Once you create  the project, clicking on its name will show you the type_id and layout_id. 

To deploy:
```
python create_tasks.py -k "AWS_ACCESS_KEY_HERE" -s "AWS_SECRET_KEY_HERE" -t "HIT_TYPE_ID" -l "HIT_LAYOUT_ID" -d "y_to_deploy_n_for_sandbox" -n NUMBER_OF_HITS > OUTPUT_TEXT_FILE
```
This will run through urls_to_names.json (a dictionary of image urls to (labelbox id, image name) tuples) and create -n HITs. The resulting HIT ids and urls used for them will be output to OUTPUT_TEXT_FILE


## get_results.py

This checks for all HITs that have been completed with the given layout_id and converts them to LabelBox JSON form 
Next it uploads the results to LabelBox

To deploy:
```
python get_results.py -k "AWS_ACCESS_KEY_HERE" -s "AWS_SECRET_KEY_HERE" -t "HIT_TYPE_ID" -l "HIT_LAYOUT_ID" -d "y_to_deploy_n_for_sandbox" -f "21" -H "OUTPUT_TEXT_FILE" 
'''

## review.py

This will run through all of the HITs and check if they have been completed on MTurk. If so it will check if they have a label stored on Labelbox. If they do the HIT's assignment will be accepted and the worker will be paid. If not, the assignment will be rejected. 
Since this approves all existing labels on labelbox and rejects all not on labelbox only run after you have ran get_results and deleted all of the bad labels from Labelbox. 
Run:
```
python review.py -k "AWS_ACCESS_KEY_HERE" -s "AWS_SECRET_KEY_HERE" -H "OUTPUT_TEXT_FILE"
```