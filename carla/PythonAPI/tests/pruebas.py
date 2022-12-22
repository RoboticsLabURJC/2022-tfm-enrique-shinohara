import sys
import glob
import os
import pandas
import cv2

import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", action='append', help="Directory to find Data")
    args = parser.parse_args()
    return args

def main():
    """args = parse_args()
    path = args.dir[0] + '*'"""
    f1 = open('data.csv', "r")
    last_line = f1.readlines()[-1]
    f1.close()
    print(last_line.split(',')[2])

if __name__ == "__main__":
    # execute only if run as a script
    main()