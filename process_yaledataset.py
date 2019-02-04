import os

dir = r"C:\Users\Kiran\Desktop\Joy\yalefaces"

# Add .png to all files in dataset so as to get to their proper format

for file in os.listdir(dir):

    if not file.endswith(".gif") and not file.endswith(".txt"):
        original_file = dir + "\\" + file
        modified_file = original_file + ".png"

        print (original_file, " --> ", modified_file)

        os.rename(original_file, modified_file)
