# -*- coding: utf-8 -*-
import json
 
images_ids = [1000]
imagesDic = {}
captionsDic = {}

# Load the json files
f = open('val.json',)
data = json.load(f)

# Change the file name to store it in
file8 = open("write_testt.txt", "w")

# Loop through the data to find the exact items we need (caption and image)
for key, value in data.items():
     for ke in value:
         for b in ke:
             if b == "caption":
                 # One image can have multiple captions
                 if ke['image_id'] not in captionsDic:
                     captionsDic[ke['image_id']] = list()
                 captionsDic[ke['image_id']].append(ke['caption'])
             if b == "file_name":
                 imagesDic[ke['id']] = ke['file_name']
                 images_ids.append(ke['id'])

# Write the data to a text file
file8.write("image,caption\n")
for j in images_ids:
    if j != 1000:
        for l in range(len(captionsDic[j])):
            file8.write("%s, %s\n" % (imagesDic[j], captionsDic[j][l]))

print('\nSummary of lengths:')
print(len(imagesDic))
print(len(captionsDic))
    
    
    
    
    
    
    