import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocess GaussianAvatars dataset with face parsing")
    parser.add_argument("--data_root", type=str, help="The root directory with the dataset's files")
    args = parser.parse_args()