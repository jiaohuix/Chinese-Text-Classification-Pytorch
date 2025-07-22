'''
python data_scripts/split_jsonl.py -i data/bert/train.jsonl -o d
ata/bert/train60w
'''
import json
import argparse
import os
from datasets import Dataset
import random

def shuffle_and_split_jsonl(input_file: str, output_dir: str, n_splits: int, seed: int = None):
    """
    Reads a JSONL file, shuffles its content, splits it into n_splits,
    and saves each split to a separate file in the output directory.

    Args:
        input_file (str): Path to the input JSONL file.
        output_dir (str): Directory to save the split files.
        n_splits (int): The number of splits to divide the data into.
        seed (int, optional): A seed for random number generation to ensure reproducibility. Defaults to None (no fixed seed).
    """
    # Set the random seed if provided
    if seed is not None:
        random.seed(seed)
        print(f"Random seed set to: {seed}")

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: '{output_dir}'")

    # 1. Read the JSONL file using datasets
    try:
        # Use streaming=True for potentially large files to avoid loading all into memory
        # If your file is manageable in memory, you can remove streaming=True
        dataset = Dataset.from_json(input_file, streaming=True)
    except Exception as e:
        print(f"Error reading JSONL file: {e}")
        return

    # Convert to a list for shuffling and splitting (if not streaming)
    # If streaming=True, we need to collect all data first.
    all_data = list(dataset)
    if not all_data:
        print("No data found in the input file.")
        return

    print(f"Successfully read {len(all_data)} records from '{input_file}'.")

    # 2. Shuffle the data
    random.shuffle(all_data)
    print("Data shuffled.")

    # 3. Calculate split sizes
    total_size = len(all_data)
    split_size = total_size // n_splits
    remainder = total_size % n_splits

    # 4. Split and save
    start_index = 0
    for i in range(n_splits):
        current_split_size = split_size + (1 if i < remainder else 0)
        end_index = start_index + current_split_size

        split_data = all_data[start_index:end_index]

        # Create a new Dataset object for the split
        split_dataset = Dataset.from_list(split_data)

        # Define output file path
        output_filename = f"split{i}.jsonl"
        output_filepath = os.path.join(output_dir, output_filename)

        # Save the split to a JSONL file
        try:
            split_dataset.to_json(output_filepath, orient="records", lines=True)
            print(f"Saved split {i} ({len(split_data)} records) to '{output_filepath}'.")
        except Exception as e:
            print(f"Error saving split {i} to '{output_filepath}': {e}")

        start_index = end_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shuffle and split a JSONL file into multiple parts with optional seed for reproducibility."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="splits",
        help="Directory to save the split files (default: 'splits')."
    )
    parser.add_argument(
        "-n", "--n_splits",
        type=int,
        default=3,
        help="The number of splits to divide the data into (default: 3)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42, # Default to None, meaning no fixed seed
        help="An integer seed for random shuffling to ensure reproducibility. If not provided, shuffling will not be reproducible."
    )

    args = parser.parse_args()

    shuffle_and_split_jsonl(args.input, args.output_dir, args.n_splits, args.seed)
