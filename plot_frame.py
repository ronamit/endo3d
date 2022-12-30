
import argparse
import glob
import os
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True,
                        help=' path to  ')
    parser.add_argument('--seq_name', type=str, required=True,
                        help='')
    parser.add_argument('--frame_time', type=float, required=True,
                        help='')
    args = parser.parse_args()


if __name__ == '__main__':
    main()
