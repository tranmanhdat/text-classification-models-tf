import os
import sys

def create_text_file(folder_in, folder_out):
    name_file = folder_in.split("/")[-1]
    f_write = open(os.path.join(folder_out, name_file+".txt"), "w+", encoding="UTF-8")
    for folder in os.listdir(folder_in):
        path_folder = os.path.join(folder_in, folder)
        for file in os.listdir(path_folder):
            with open(os.path.join(path_folder, file), "r", encoding="UTF-8") as f_read:
                text = f_read.read()
                f_write.write("{}\t{}\n".format(folder, text))
    f_write.close()
root = sys.argv[1]
out_dir = sys.argv[2]
train_path = os.path.join(root, "train")
test_path = os.path.join(root, "test")
create_text_file(train_path, out_dir)
create_text_file(test_path, out_dir)