import os

class_path = 'data/anchor'

# Extract classes from path
classes = os.listdir(class_path)
sorted_class = sorted(classes)
print(sorted_class)

# Create an empty dictionary
class_indices = {}
print(dict(enumerate(sorted_class)))

# Rearrange the placement of username and index
for index, username in enumerate(sorted_class):
    class_indices[username] = index
    # class_indices.update({username: index})

print(class_indices)