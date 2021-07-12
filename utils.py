import argparse


def image_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Image file path")
    image_args = vars(ap.parse_args())
    return image_args
