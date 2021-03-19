import os
import sys

def get_alphabet(path_file, file_name):
    f_write = open(file_name, "w+", encoding="UTF-8")
    f_read = open(path_file, "r", encoding="UTF-8")
    characters = []
    for line in f_read:
        elements = line.rstrip().split("\t")
        for character in elements[1]:
            if character not in characters:
                characters.append(character)
    f_write.write("".join(characters))
    f_write.close()
file_path = sys.argv[1]
file_out = sys.argv[2]
get_alphabet(file_path, file_out)