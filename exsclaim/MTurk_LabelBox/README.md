# Setting up Labeling Tasks with Mechanical Turk and LabelBox

## create_tasks.py

Used to create HITs for a specified project on Mechanical Turk.
Project must first be created as a requester at https://requester.mturk.com/create/projects/new
Select "Other", at the very bottom left. 
Select sign in to create project, sign in. 
Enter your desired properties. These determine type_id
In the Design Layout tab, delete the text that is there and paste the text of working_layout.xml there. This determines layout_id.
Once you create  the project, clicking on its name will show you the type_id and layout_id. 

## get_results.py

This checks for all HITs that have been completed with the given layout_id and converts them to LabelBox JSON form 
Next it uploads the results to LabelBox