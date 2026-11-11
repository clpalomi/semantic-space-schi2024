# Additional analysis; check additional/README.md before running.
from pipeline.paths import ADDITIONAL_RESULTS, SUMMARIES

# -*- coding: utf-8 -*-
"""
@author: Claudio
"""

# pip install pandas openpyxl

import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font

# PATHS
path_input = SUMMARIES
path_results = ADDITIONAL_RESULTS

# List of all the xlsx files in the folder
files = [f for f in os.listdir(path_input) if f.endswith('.xlsx')]

# Create a new Excel workbook
wb = Workbook()
ws = wb.active

# Iterate through the files
for file in files:
    df = pd.read_excel(os.path.join(path_input, file))
    # Write name of the file in bold
    cell = ws.cell(row=ws.max_row+1, column=1, value=file[:-5])
    cell.font = Font(bold=True)
    # Write the headers
    ws.append(list(df.columns))
    for index, row in df.iterrows():
        ws.append(row.tolist())

# Save the workbook
output_file = path_results+'all_summaries.xlsx'
wb.save(output_file)
