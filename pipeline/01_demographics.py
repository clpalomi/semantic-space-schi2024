from pipeline.paths import METADATA

# -*- coding: utf-8 -*-
"""
@author: Claudio
"""

import pandas as pd

# PATH INPUT
path_demo = METADATA

# IMPORT DEMOGRAPHICS
groups = pd.read_csv(path_demo+'ssd_mdd_hc_data.csv')

# CHANGE NAMES
column_names = {'Proband': 'name', 'Typ2': 'group', 'Typ': 's_group', 'Alter': 'age',
                'SKID_Diagnose_1': 'type',
                'Geschlecht': 'gender', 'Bildungsjahre': 'years_ed'}
groups = groups.rename(columns=column_names)
groups = groups[['name', 'group', 's_group', 'type']]

# TO EXCEL
groups.to_excel(path_demo+'groups.xlsx', index=False)


