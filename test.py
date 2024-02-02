import pandas as pd
import os
import re

def extract_metrics(file_path):
    """ Extracts the MRR test result and Hits@1 to Hits@100 from a given log file. """
    metrics = {}
    found_metrics = False  # Flag to check if any metric is found
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if "MRR result:" in line:
                    parts = line.split(',')
                    metrics['MRR Test Result'] = parts[-1].strip().replace("Test: ", "")
                    found_metrics = True
                elif "Hits@" in line:
                    hit_match = re.search(r"Hits@(\d+) result:.*?Test: (\d+\.\d+ ± \d+\.\d+)", line)
                    if hit_match:
                        hits_key = f"Hits@{hit_match.group(1)} Test Result"
                        hits_value = hit_match.group(2)
                        metrics[hits_key] = hits_value
                        found_metrics = True
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    if not found_metrics:
        print(f"No metrics found in file: {file_path}")
    return metrics

def extract_relevant_part_of_filename(file_name):
    """ Extracts the relevant part of the file name. """
    parts = file_name.split('_', 2)  # Splitting at most 2 times
    if len(parts) > 2:
        return parts[2]  # Return the part after the second underscore
    return file_name

# Path to the folder containing the log files
folder_path = 'Logs/LM_GNN_train/citeseer/sbert/True'

# List all log files in the folder
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.log')]

if not file_paths:
    print("No log files found in the folder.")

# Extracting metrics
all_metrics = []
for file_path in file_paths:
    metrics = extract_metrics(file_path)
    if metrics:
        file_name = os.path.basename(file_path)
        metrics['File Name'] = extract_relevant_part_of_filename(file_name)
        all_metrics.append(metrics)

# Creating a DataFrame
df = pd.DataFrame(all_metrics)

# Check if DataFrame is empty
if df.empty:
    print("No data extracted for DataFrame.")
else:
    print(df)
