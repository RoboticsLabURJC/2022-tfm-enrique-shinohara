import sys
import glob
import os
import pandas
import cv2

import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', help="Directory to find Data")
    args = parser.parse_args()
    return args

def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, "#"*x, "."*(size-x), j, count), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def main():
    args = parse_args()
    data_dir = args.data_dir[0]

    list_images = glob.glob(data_dir + '*')
    data_csv_name = "data.csv"
    csv_dir = data_dir + data_csv_name
    data_csv = pandas.read_csv(csv_dir)

    deleted = False

    for i in progressbar(range(0, len(list_images)), "Computing: ", 40):
        i = list_images[i]
        if 'data' in i:
            continue
        img = cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if (7 in img[:,:,0]) and (0 in img[:,:,1]) and (0 in img[:,:,2]):
            # print(i.split('/')[1])
            deleted = True
            data_csv.drop(data_csv[data_csv['image'] == str(i.split('/')[1])].index, inplace=True)
            os.remove(i)
    if deleted:
        print("IMAGES DELETED")
    else:
        print("NO IMAGES TO DELETE")

    data_csv.to_csv(csv_dir, index=False)

if __name__ == "__main__":
    # execute only if run as a script
    main()