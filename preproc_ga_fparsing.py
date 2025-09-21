import argparse
import face_parsing
import scene

k_chkp_path = "./face_parsing/chkp/79999_iter.pth"

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocess GaussianAvatars dataset with face parsing")
    parser.add_argument("--data_root", type=str, help="The root directory with the dataset's files")
    parser.add_argument("--dst_path", type=str, help="The path to save the processed data.")
    args = parser.parse_args()

    dataset = scene.SceneGaussianAvatars(args.data_root, 'transforms_train.json')



    face_parsing.evaluate()