# run_pipeline.py
import argparse
from pathlib import Path
from pipeline import fit_transform_train, transform_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="CMAPSSData/raw", help="Raw data directory containing train/test/RUL files")
    args = parser.parse_args()

    train_path = Path(args.raw_dir) / "train_FD001.txt"
    rul_path = Path(args.raw_dir) / "RUL_FD001.txt"
    test_path = Path(args.raw_dir) / "test_FD001.txt"

    print("Running pipeline fit_transform_train ...")
    fit_transform_train(str(train_path), str(rul_path))
    print("Transforming test set ...")
    transform_test(str(test_path))
    print("Pipeline complete — artifacts saved in ./pipeline_outputs/")