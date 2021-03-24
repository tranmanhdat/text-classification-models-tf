from data_utils import *
import sys


NUM_CLASS = 12
BATCH_SIZE = 64
NUM_EPOCHS = 10
WORD_MAX_LEN = 100
CHAR_MAX_LEN = 256

print("Building dataset...")
path_train = sys.argv[1]
x, y, alphabet_size, le = build_char_dataset(path_train, CHAR_MAX_LEN)