import os
import sys

target_path=""

file_list = os.listdir(target_path+"/image")
file_list.sort()

print(len(file_list))

with open(target_path+"/train.txt", "w") as txtfile:

    for entry in file_list:
        txtfile.write(entry.rstrip(".jpg") + "\n")

# check for unmatched pairs:
list1=[file.split(".")[0] for file in os.listdir(target_path+"/image")]
list2= [file.split(".")[0] for file in os.listdir(target_path+"/mask")]

# check for missing parts
set1 = set(list1)
set2 = set(list2)

unique_to_list1 = set1 - set2
unique_to_list2 = set2 - set1

print(unique_to_list1)
print(unique_to_list2)