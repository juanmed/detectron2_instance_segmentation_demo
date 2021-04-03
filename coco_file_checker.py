import json

#Find JSON that gives errors
JSON_LOC="train2.json"

#Open JSON
val_json = open(JSON_LOC, "r")
json_object = json.load(val_json)
val_json.close()


print("First check...")
idx = []
for i, instance in enumerate(json_object["annotations"]):
    if len(instance["segmentation"][0]) <= 4:
        print("instance number", i, "raises arror:", instance["segmentation"][0])
        idx.append(i)

print("Replacing... ")
for i in idx:
	#Alter object generating the error with something random not causing the error
	org = json_object["annotations"][i]["segmentation"][0]
	org.append(min(0,org[0]-1))
	org.append(min(0,org[1]-1))
	json_object["annotations"][1030]["segmentation"] = [org]

print("Second check")
for i, instance in enumerate(json_object["annotations"]):
    if len(instance["segmentation"][0]) <= 4:
        print("instance number", i, "raises arror:", instance["segmentation"][0])
        idx.append(i)


print("done")
#Write back altered JSON
#val_json = open("train2.json", "w")
#json.dump(json_object, val_json)
#val_json.close()