import io
import os
import unicodedata
import requests
import string
import glob
import torch
import random
from zipfile import ZipFile


ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)


# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in ALL_LETTERS
    )


def load_data():
    if not os.path.exists("./data/data.zip"):
        print("Downloading data")
        r = requests.get("https://download.pytorch.org/tutorial/data.zip")
        with open("./data/data.zip", "wb") as f:
            f.write(r.content)

        with ZipFile("./data/data.zip", "r") as zip:
            zip.extractall()
    print("Data downloaded, reading files")

    category_lines = {}
    all_categories = []

    # Read a file and split into lines
    def read_lines(filename):
        lines = io.open(filename, encoding="utf-8").read().strip().split("\n")
        return [unicode_to_ascii(line) for line in lines]

    for filename in glob.glob("data/names/*.txt"):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)

        lines = read_lines(filename)
        category_lines[category] = lines

    return category_lines, all_categories


"""
To represent a single letter, we use a “one-hot vector” of
size <1 x n_letters>. A one-hot vector is filled with 0s
except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 0 ...>.
To make a word we join a bunch of those into a
2D matrix <line_length x 1 x n_letters>.
That extra 1 dimension is because PyTorch assumes
everything is in batches - we’re just using a batch size of 1 here.
"""


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor


if __name__ == "__main__":
    print(ALL_LETTERS)
    print(unicode_to_ascii("Ślusàrski"))

    category_lines, all_categories = load_data()
    print(category_lines["Italian"][:5])

    print(letter_to_tensor("J"))  # [1, 57]
    print(line_to_tensor("Jones").size())  # [5, 1, 57]

    load_data()
