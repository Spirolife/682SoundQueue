# PURPOSE OF FILE: Utility functions for the soundqueue project. 
# Mainly includes printing data, saving data/graphs, and displaying graphs

import matplotlib.pyplot as plt
import torch
import os

# Macros we may need for all files
track_metadata = {}
genre_data = {}

# Shows a plot and saves it to the specified path
def show_wait_destroy(item):
    # check if item is an image or a plot
    if isinstance(item, torch.Tensor):
        # item is an image
        plt.imshow(item)

    plt.show()
    plt.waitforbuttonpress()
    plt.close()

# Example format for calling this function:
# diagnostic_print("!" + "Error message")
def diagnostic_print(message):
    assert_val(False, message)

def assert_val(val, message):
    if not val:
        if "!" in message:
            # remove the "!" from the message

            
            ColorPrint.printerror(message[1:])
        elif "?" in message:
            ColorPrint.printwarning(message[1:])
        elif "*" in message:
            ColorPrint.printsuccess(message[1:])
        elif "#" in message:
            ColorPrint.printinfo(message[1:])
        else:
            ColorPrint.printbold(message)

class ColorPrint:
    def printerror(message):
        # red
        print("\033[91m {}\033[00m" .format(message))
    def printwarning(message):
        # yellow
        print("\033[93m {}\033[00m" .format(message))
    def printsuccess(message):
        # green
        print("\033[92m {}\033[00m" .format(message))
    def printinfo(message):
        # blue
        print("\033[94m {}\033[00m" .format(message))
    def printbold(message):
        # bold
        print("\033[1m {}\033[00m" .format(message))