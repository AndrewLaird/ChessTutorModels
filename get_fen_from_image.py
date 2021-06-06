# assume this is run after detect.py has been run, this means that the images in data/images
# have corresponding data in labels
from PIL import Image
import numpy as np
import pandas as pd
import os
import random
import sklearn
import skimage
import skimage.io
import matplotlib.pyplot as plt
import detect
import time
import pathlib
import sys

PROJECT_DIR = str(pathlib.Path(__file__).parent.absolute()) +'/'
TO_CROP_LABELS_DIR =  PROJECT_DIR + 'data/cropped/labels'
CROPPED_DIR  = PROJECT_DIR + 'data/cropped'
OUTPUT_DIR = PROJECT_DIR + 'data/output_files'

CROP_MODEL = PROJECT_DIR + 'cropModel.pt'
PIECE_MODEL = PROJECT_DIR + 'pieceModel.pt'

# get command line arguments
_, INPUT_DIR, upload_number = sys.argv
CROPPED_DIR = CROPPED_DIR+'/'+ upload_number

if(not os.path.exists(CROPPED_DIR)):
    os.mkdir(CROPPED_DIR)

# rename input files to be 1->N
detected_files = []
for (dirpath, dirnames, filenames) in os.walk(INPUT_DIR):
    for i,file in enumerate(filenames):
        # convert not png to png
        if(file.split(".")[-1] != '.png'):
            im = Image.open(INPUT_DIR + '/' + file)
            im.save(INPUT_DIR + '/' + file)
            detected_files.append(file)
         except Exception as e:
             print("couldn't open file")
             print(e);
    break


# start off by running the crop model on the input images
crop_data = detect.detect(INPUT_DIR, CROP_MODEL, CROPPED_DIR)

def crop_image(file,possible_chessboards, output_filename, output_dir):
    width, height = image.size
    # find the most confident of the chessboards here
    # but for now just get the first one
    most_confident_chessboard = sorted(possible_chessboards, key=lambda row: row[5])[0]
    class_id, center_x, center_y,  box_width, box_height, confidence = most_confident_chessboard
    pixel_center_x = width*center_x
    pixel_center_y = height*center_y
    pixel_box_width = width*box_width
    pixel_box_height = height*box_height
    box_top_left_x = pixel_center_x - pixel_box_width/2
    box_top_left_y = pixel_center_y - pixel_box_height/2



    left = box_top_left_x
    top= box_top_left_y
    right =  (box_top_left_x + pixel_box_width )
    bottom = (box_top_left_y + pixel_box_height )
    cropped_image = image.crop((left, top, right, bottom))
    cropped_image = cropped_image.resize((1200,1200))

    #image.show()
    cropped_image.save(output_dir + "/" + output_filename)
    return list(most_confident_chessboard)

crops = []
for i in range(len(detected_files)):
    image = Image.open(INPUT_DIR + '/' + detected_files[i])
    chessboard = crop_image(image,crop_data[i],detected_files[i],CROPPED_DIR)
    crops.append(chessboard)


# then we run those cropped images through the piece model
detected_pieces = detect.detect(CROPPED_DIR, PIECE_MODEL, OUTPUT_DIR)


# this outputs files that we can process into the final chess board!
fen_array  = ['P','N','B','R','K','Q','p','n','b','r','k','q']
def board_to_fen(board):
    fen_str = ''
    for y in range(8):
        if(y>0):
            fen_str += '/'
        row = board[y]
        counter = 0
        row_str = ""
        for x in range(8):
            piece = row[x]
            if(piece == 12):
                counter += 1
            else:
                if(counter != 0):
                    row_str += str(counter)
                    counter = 0
                row_str+= fen_array[int(piece)]
        if(counter != 0):
            row_str += str(counter)
        fen_str += row_str

    #who's turn is it
    # fen_str += " w"

    #who can castle and what was the last move
    # fen_str += ' - -'
    return fen_str

fens = []

for i in range(len(detected_files)):
    # intialize the boards all empty
    combined_confidence = 0
    total_number_of_pieces = 0
    board = [[12 for x in range(8)] for y in range(8)]
    for piece in detected_pieces[i]:
        class_id, center_x, center_y,  box_width, box_height, confidence = piece
        if(confidence < .85):
            continue
        combined_confidence += confidence
        # split the center_x/y into the 8th it belongs in
        x_pos = int(center_x // (.125))
        y_pos = int(center_y // (.125))
        board[y_pos][x_pos] = class_id
        total_number_of_pieces += 1
    fens.append(board_to_fen(board))

if(len(fens) == 1):
    with open(OUTPUT_DIR+"/"+upload_number +".txt", "w+") as f:
        f.write("0\n")
        f.write(" ".join([str(x) for x in crops[0]])+"\n")
        f.write(fens[0]+"\n")
    exit(0)
else:
    with open(OUTPUT_DIR+"/"+upload_number +".txt", "w+") as f:
        f.write("1\n")
    exit(0)
