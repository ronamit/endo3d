import argparse
from util import get_frame_at_timestamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ffmpeg_path', type=str, required=True, help='path to ffmpeg executable')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help=' path to  the dataset to be loaded')
    parser.add_argument('--seq_name', type=str, required=True,
                        help='name of sequence to be loaded')
    parser.add_argument('--frame_time', type=float, required=True,
                        help='the in-sequence time of the frame to load [seconds]')
    args = parser.parse_args()

    vid_file_name = args.seq_name + '_RGB'
    get_frame_at_timestamp(args.dataset_path, args.seq_name, args.frame_time, vid_file_name)


if __name__ == '__main__':
    main()
