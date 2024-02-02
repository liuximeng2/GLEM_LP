import pandas as pd
import os
import re

def extract_mrr_results(file_path):
    """ Extracts the training and test MRR results from a given log file. """
    results = {'Training MRR Result': None, 'Test MRR Result': None}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if "MRR result:" in line:
                    # Extracting training and test MRR results
                    train_match = re.search(r"Train: (\d+\.\d+ ± \d+\.\d+)", line)
                    test_match = re.search(r"Test: (\d+\.\d+ ± \d+\.\d+)", line)
                    if train_match:
                        results['Training MRR Result'] = train_match.group(1)
                    if test_match:
                        results['Test MRR Result'] = test_match.group(1)
                    break  # Assuming only one line contains MRR results
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return results

def extract_relevant_part_of_filename(file_name):
    """ Extracts the relevant part of the file name. """
    parts = file_name.split('_', 2)  # Splitting at most 2 times
    if len(parts) > 2:
        return parts[2]  # Return the part after the second underscore
    return file_name  # Return the original name if the format is different

# Path to the folder containing the log files
folder_path = 'Logs/LM_GNN_train/pubmed/sbert/True'  # Replace with the path to your folder

# List all log files in the folder
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.log')]

if not file_paths:
    print("No log files found in the folder.")

# Extracting MRR results
all_results = []
for file_path in file_paths:
    mrr_results = extract_mrr_results(file_path)
    if mrr_results:
        file_name = os.path.basename(file_path)
        relevant_file_name = extract_relevant_part_of_filename(file_name)
        all_results.append({
            'File Name': relevant_file_name,
            'Training MRR Result': mrr_results['Training MRR Result'],
            'Test MRR Result': mrr_results['Test MRR Result']
        })

# Creating a DataFrame
df = pd.DataFrame(all_results)

# Check if DataFrame is empty
if df.empty:
    print("No data extracted for DataFrame.")
else:
    print(df)
