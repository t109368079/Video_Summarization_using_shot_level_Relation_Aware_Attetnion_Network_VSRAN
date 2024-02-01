import json
import random

def generate_splits(sequence, number_of_splits, shuffle):
    segments = even_partition(sequence, number_of_splits, shuffle = True)
    splits = []
    for validate_segment in segments:
        splits.append({'train': flat([segment for segment in segments if segment != validate_segment]), 'validate': validate_segment})
    return splits


def even_partition(sequence, number_of_segments, shuffle):
    if shuffle is True:
        random.shuffle(sequence)
    return [sequence[index::number_of_segments] for index in range(number_of_segments)]


def flat(sequence):
    return [element for subsequence in sequence for element in subsequence]


def read_splits(path):
    splits_file = open(path, 'r', encoding = 'utf-8')
    splits_string = splits_file.read()
    splits = json.loads(splits_string)
    return splits


def write_splits(splits, path):
    splits_file = open(path, 'w', encoding = 'utf-8')
    json.dump(splits, splits_file)
    return


if __name__ == '__main__':
    pass
