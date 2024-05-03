import re

# Read the file
with open("test_binds.txt", "r") as file:
    data = file.read()

# Define regex pattern to match numbers and associated dot
pattern = re.compile(r"\b\d+\. ", re.MULTILINE)

# Remove the numbers and associated dot
cleaned_data = re.sub(pattern, "", data)

# Write the cleaned data to a new file
with open("cleaned_test_binds.txt", "w") as file:
    file.write(cleaned_data)
