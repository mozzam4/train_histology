import itertools
import multiprocessing

from openslide.deepzoom import DeepZoomGenerator
import openslide
import os
from skimage import io, color, feature
import numpy as np
from scipy import ndimage as ndi
import staintools
import random
import pandas
import sys
import argparse
from pathlib import Path
import json
import sys
from p_tqdm import p_map
def create_arg_parser():
    # Creates and returns the ArgumentParser object
    parser = argparse.ArgumentParser(description='Train arguments')
    parser.add_argument('inputDirectory', help='Path to the input directory.')
    return parser


def get_max_zoom_level(matched):
    max_level = 999
    for each in matched:
        input_filename = each['Path']
        img = openslide.OpenSlide(input_filename)
        tiles = DeepZoomGenerator(img, tile_size=512, overlap=0, limit_bounds=False)
        if tiles.level_count < max_level:
            max_level = tiles.level_count
    max_level = max_level - 1 #index starts with 0
    return max_level


def process_one_image(each):
    input_filename = each['Path']
    output_filedir = os.path.join(out_dir, input_filename[82:105])
    if not os.path.isdir(output_filedir):
        # Create a new directory because it does not exist
        os.makedirs(output_filedir)

    img = openslide.OpenSlide(input_filename)
    tiles = DeepZoomGenerator(img, tile_size=512, overlap=0, limit_bounds=False)
    cols, rows = tiles.level_tiles[desired_level]
    total_random_sample = random.sample(set(itertools.product(list(range(cols)), list(range(rows)))), cols * rows)
    no_patches = 0
    for i in total_random_sample:
        if no_patches < 500:
            col, row = i
            tile = tiles.get_tile(desired_level, (col, row))
            tile = np.asanyarray(tile)
            if tile.mean() < 230 and tile.std() > 15:
                # To remove background tiles and tiles with very less tissue data
                tile_gray = color.rgb2gray(tile)
                tile_blur = ndi.gaussian_filter(tile_gray, 3)
                edges1 = feature.canny(tile_blur, sigma=1)
                unique, counts = np.unique(edges1, return_counts=True)
                if len(unique) < 2:  # (to avoid full white)
                    continue

                edge_to_image_ratio = counts[1] / counts[0]

                if edge_to_image_ratio >= 0.01:  # To remove blur tiles

                    # to_transform = staintools.read_image(input_file)
                    # Standardize brightness (optional, can improve the tissue mask calculation)
                    src = staintools.LuminosityStandardizer.standardize(target)
                    to_transform = staintools.LuminosityStandardizer.standardize(tile)

                    # Stain normalize
                    normalizer = staintools.StainNormalizer(method='macenko')
                    normalizer.fit(src)
                    transformed = normalizer.transform(to_transform)

                    tile_filename = input_filename[82:105] + '_' + '(' + str(col) + ',' + str(
                        row) + ')'
                    tile_filename = os.path.join(output_filedir, tile_filename) + '.jpg'
                    io.imsave(tile_filename, transformed)
                    no_patches = no_patches + 1
        else:
            break

    #filenames.append(input_filename[82:105])
    #patches.append(no_patches)
    #MSI_stats.append(each['Label'])
    return input_filename[82:105], no_patches, each['Label']


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    if not os.path.exists(parsed_args.inputDirectory):
        print("Input file path does not exist")
        exit()

    out_dir = os.path.join(parsed_args.inputDirectory, Path('Few_patches/sel'))
    ref_file = os.path.join(parsed_args.inputDirectory, Path('Few_patches/Ref.png'))
    out_annotation_file = os.path.join(parsed_args.inputDirectory, Path('Few_patches/annotations.csv'))
    matched_file_path = os.path.join(parsed_args.inputDirectory, Path('matched.json'))

    target = staintools.read_image(ref_file)
    # filenames = []
    # patches = []
    # MSI_stats = []
    with open(matched_file_path, "r") as final:
        matched = json.load(final)
    desired_level = get_max_zoom_level(matched)
    #pool = multiprocessing.Pool()
    #return_values = process_one_image(matched[0])
    return_values = p_map(process_one_image, matched)
    filenames, patches, MSI_stats = zip(*return_values)
    #print(return_values)
    #filenames, patches, MSI_stats = pool.map(process_one_image, matched)
    # for each in matched:
    #     input_filename = each['Path']
    #     output_filedir = os.path.join(out_dir, input_filename[82:105])
    #     if not os.path.isdir(output_filedir):
    #         # Create a new directory because it does not exist
    #         os.makedirs(output_filedir)
    #
    #     img = openslide.OpenSlide(input_filename)
    #     tiles = DeepZoomGenerator(img, tile_size=512, overlap=0, limit_bounds=False)
    #     cols, rows = tiles.level_tiles[tiles.level_count-3]
    #     total_random_sample = random.sample(set(itertools.product(list(range(cols)), list(range(rows)))), cols*rows)
    #     no_patches = 0
    #
    #     for i in total_random_sample:
    #         if no_patches < 500:
    #             col, row = i
    #             tile = tiles.get_tile(tiles.level_count-3, (col, row))
    #             tile = np.asanyarray(tile)
    #             if tile.mean() < 230 and tile.std() > 15:
    #                 # To remove background tiles and tiles with very less tissue data
    #                 tile_gray = color.rgb2gray(tile)
    #                 tile_blur = ndi.gaussian_filter(tile_gray, 3)
    #                 edges1 = feature.canny(tile_blur, sigma=1)
    #                 unique, counts = np.unique(edges1, return_counts=True)
    #                 if len(unique) < 2:  #(to avoid full white)
    #                     continue
    #
    #                 edge_to_image_ratio = counts[1] / counts[0]
    #
    #                 if edge_to_image_ratio >= 0.01:  # To remove blur tiles
    #
    #                     # to_transform = staintools.read_image(input_file)
    #                     # Standardize brightness (optional, can improve the tissue mask calculation)
    #                     src = staintools.LuminosityStandardizer.standardize(target)
    #                     to_transform = staintools.LuminosityStandardizer.standardize(tile)
    #
    #                     # Stain normalize
    #                     normalizer = staintools.StainNormalizer(method='macenko')
    #                     normalizer.fit(src)
    #                     transformed = normalizer.transform(to_transform)
    #
    #                     tile_filename = input_filename[82:105] + '_' + '(' + str(col) + ',' + str(
    #                         row) + ')'
    #                     tile_filename = os.path.join(output_filedir, tile_filename) + '.jpg'
    #                     io.imsave(tile_filename, transformed)
    #                     no_patches = no_patches + 1
    #         else:
    #             break
    #
    #     filenames.append(input_filename[82:105])
    #     patches.append(no_patches)
    #     MSI_stats.append(each['Label'])

    data = {'name': filenames, 'no_of_files': patches, 'if_msi': MSI_stats}
    df = pandas.DataFrame(data=data)
    df.to_csv(out_annotation_file, sep='\t', encoding='utf-8', index=False)
