import boto3

endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'


# for command line usage
ap = argparse.ArgumentParser()
ap.add_argument("-k", "--access_key", type=str,
                help="AWS Access Key ID")
ap.add_argument("-s", "--secret_key", type=str,
                help="AWS Secret Access Key")

args = vars(ap.parse_args())


# parse command line arguments
access_key = args["access_key"]
secret_key = args["secret_key"]


mtc = boto3.client('mturk', aws_access_key_id=access_key,
					   aws_secret_access_key=secret_key,
					   region_name='us-east-1', 
					   endpoint_url = endpoint_url)
                       

hit_ids = []

res = mtc.list_hits(MaxResults=100)
next = res["NextToken"]
i = res["NumResults"]
while True:
    res = mtc.list_hits(MaxResults=100, NextToken=next)     
    i += res["NumResults"]
    hit_ids += [res["HITs"][i]["HITId"] for i in range(len(res["HITs"]))]
    if "NextToken" in res:
        next = res["NextToken"]
    else:
        break


rejected = {}
accepted = {}
pending = {}
print("There are {} HITs".format(len(hit_ids)))

for hitid in hit_ids:
    res = mtc.list_assignments_for_hit(HITId = hitid)
    for assignment in res["Assignments"]:
        status = assignment["AssignmentStatus"]
        worker = assignment["WorkerId"]
        id = assignment["AssignmentId"]
        if status == "Submitted":
            pend = pending.get(worker, [])
            pend.append(id)
            pending[worker] = pend
        elif status == "Approved":
            pend = accepted.get(worker, [])
            pend.append(id)
            accepted[worker] = pend           
        elif status == "Rejected":
            pend = rejected.get(worker, [])
            pend.append(id)
            rejected[worker] = pend  
        else:
            print("error {}".format(hitid))
            
workers = set(rejected.keys())
workers.update(set(pending.keys()))
workers.update(set(accepted.keys()))

total_approved = 0
total_rejected = 0
totoal_outstanding = 0


for worker in workers:
    good = len(accepted.get(worker, []))
    bad = len(rejected.get(worker, []))
    ugly = len(pending.get(worker, []))
    total_approved += good
    total_rejected += bad
    totoal_outstanding += ugly
    print("{} has {} approved, {} rejected, {} outstanding".format(worker, good, bad, ugly))
    
print("TOTAL: {} approved, {} rejected, {} outstanding".format(total_approved, total_rejected, totoal_outstanding))
