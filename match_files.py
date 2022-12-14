
import pandas
import os
import argparse
from pathlib import Path
import sys
import json

def create_arg_parser():
    # Creates and returns the ArgumentParser object
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('inputDirectory', help='Path to the input directory.')
    parser.add_argument('inputfile', help='Path to the input csv file.')
    parser.add_argument('matched_file_path', help='Path to store matched file and labels in json.')
    return parser


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])

    in_dir = parsed_args.inputDirectory
    in_file = parsed_args.inputfile
    all_files = []
    matched_files = []

    fields = ['Sample ID', 'MSI Status']
    for filename in os.listdir(in_dir):
        file = os.path.join(in_dir, filename)
        my_file = open(file, "r")
        data = my_file.read()
        data_into_list = data.split("\n")
        data_into_list.pop()
        #data = [s[43:58] for s in data_into_list]
        all_files.extend(data_into_list)
        my_file.close()

    in_data_file = pandas.read_csv(in_file, encoding='utf-8', skipinitialspace=True, usecols=fields)
    source_list = in_data_file['Sample ID'].to_list()
    msi_list = in_data_file['MSI Status'].to_list()
    for ind, item in enumerate(source_list):
        indices = [i for i, s in enumerate(all_files) if item in s]
        if not indices == []:
            index = indices[0]
            label = 0
            if (msi_list[ind] == 'MSS') or (msi_list[ind] == 'MSI-L'):
                label = 0
            else:
                label = 1
            match_dict = {'Path': all_files[index], 'Label': label}
            matched_files.append(match_dict)

    with open(parsed_args.matched_file_path, "w") as final:
        json.dump(matched_files, final)

    with open(parsed_args.matched_file_path, "r") as final:
        matched = json.load(final)
    print(matched)

