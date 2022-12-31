import argparse
from matplotlib import pyplot as plt

from util import get_frame_at_timestamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True,
                        help=' path to  the dataset to be loaded')
    parser.add_argument('--seq_name', type=str, required=True,
                        help='name of sequence to be loaded')
    parser.add_argument('--frame_time', type=float, required=True,
                        help='the in-sequence time of the frame to load [seconds]')
    args = parser.parse_args()

    vid_file_name = args.seq_name + '_RGB'
    rgb_frame, _ = get_frame_at_timestamp(args.dataset_path, args.seq_name, args.frame_time, vid_file_name)
    plt.imshow(rgb_frame)
    plt.show()

    depth_vid_file_name = args.seq_name + '_Depth'
    depth_frame, _ = get_frame_at_timestamp(args.dataset_path, args.seq_name, args.frame_time, depth_vid_file_name)
    plt.imshow(depth_frame)
    plt.show()


if __name__ == '__main__':
    main()
