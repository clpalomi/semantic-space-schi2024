"""Identical text helpers from the supplied scripts."""

import os
import re
import numpy as np
import pandas as pd

def calculate_diffs(sem_list):
    # Calculate first differences
    differences = [sem_list[i] - sem_list[i - 1] for i in range(1, len(sem_list))]
    # mean and std
    m_diff = np.mean(differences)
    s_diff = np.std(differences)
    # Calculate absolute values
    absolute_values = [abs(diff) for diff in differences]
    # Calculate sum of differences
    sum_differences = sum(differences)
    # Calculate sum of absolute values
    sum_absolute_values = sum(absolute_values)
    # Calculate the mean of absolute values (amplitude change rate)
    mean_absolute_values = sum_absolute_values / len(absolute_values) if absolute_values else 0
    return [differences, m_diff, s_diff,
            absolute_values, sum_differences, 
            sum_absolute_values, mean_absolute_values]

def crossings_average(sem_list):
    # Calculate the average value of the list
    average_value = np.mean(sem_list)
    # Initialize counter for crossings
    crossings_count = 0
    # Iterate through consecutive pairs of values, starting from the second element
    for i in range(1, len(sem_list)):
        current_value = sem_list[i]
        previous_value = sem_list[i - 1]
        # Check if one value is above average and the other is not
        if (current_value > average_value and previous_value <= average_value) or \
           (previous_value > average_value and current_value <= average_value):
            crossings_count += 1
    return [crossings_count, crossings_count/(len(sem_list)-1)]

def length_statistics(sentence_lengths):
    data = {'filename': [], 'total_length': [], 'sent_av_length': [], 'sent_std_length': []}
    for filename, lengths in sentence_lengths.items():
        total_length = sum(lengths)
        average_length = np.mean(lengths)
        std_length = np.std(lengths)
        data['filename'].append(filename)
        data['total_length'].append(total_length)
        data['sent_av_length'].append(average_length)
        data['sent_std_length'].append(std_length)
    df = pd.DataFrame(data)
    return df

def read_files_sent(folder_path):
    files_dict = {}
    sentence_pattern = r'[^.!?]*[.!?]'  # Pattern to identify sentences ending with '.', '!', or '?'
    filename_pattern = r'^.*bild_\d+'
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if re.search(filename_pattern, filename) and filename.endswith(".txt"):
                print(filename)
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read().strip()
                    # Add '.' at the end if not present
                    if text[-1] not in '.!?':
                        text += '.'
                    # Replace newline characters with '.' after the last word
                    text = re.sub(r'(\S)\s*\n', r'\1.', text)
                    sentences = re.findall(sentence_pattern, text)
                    # Remove leading whitespaces from each sentence
                    sentences = [sentence.strip() for sentence in sentences]
                    files_dict[filename] = sentences
    return files_dict

def sentence_lengths(text_files):
    lengths_dict = {}
    for filename, sentences in text_files.items():
        lengths_per_filename = []
        for sentence in sentences:
            words = sentence.split()  # Split sentence into words
            length = len(words)
            lengths_per_filename.append(length)
        lengths_dict[filename] = lengths_per_filename
    return lengths_dict

def slope_sign_changes(data):
    ssc = 0
    for i in list(range(len(data)-1))[1:]:
        if (data[i] < data[i+1]) and (data[i] < data[i-1]):
            ssc += 1
        elif (data[i] > data[i+1]) and (data[i] > data[i-1]):
            ssc += 1
        else:
            ssc += 0
    return ssc
