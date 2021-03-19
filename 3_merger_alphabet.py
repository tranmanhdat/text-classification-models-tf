import os, sys

def get_alphabet(path_file, file_name):
    characters = []
    for file in path_file:
        f_write = open(file_name, "w+", encoding="UTF-8")
        f_read = open(file, "r", encoding="UTF-8")
        for character in f_read.read():
            if character not in characters:
                characters.append(character)
    f_write.write("".join(characters))
    f_write.close()
path_file = []
path_file.append(sys.argv[1])
path_file.append(sys.argv[2])
file_out = sys.argv[3]
get_alphabet(path_file, file_out)