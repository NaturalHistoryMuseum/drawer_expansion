import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "/Users/mbax9qg2/Downloads/drawer/aim123/specimen_results.csv"
chunk_size = 100000  # Adjust based on available memory
chunks = pd.read_csv(path, chunksize=chunk_size)
df = pd.concat(chunks, ignore_index=True)
print(f"Final DataFrame shape of Specimen CSV: {df.shape}")

path = "/Users/mbax9qg2/Downloads/drawer/aim123/drawer_results.csv"
chunk_size = 100000  # Adjust based on available memory
chunks = pd.read_csv(path, chunksize=chunk_size)
df1 = pd.concat(chunks, ignore_index=True)
print(f"Final DataFrame shape of Drawer CSV: {df1.shape}")

path = "/Users/mbax9qg2/Downloads/drawer/aim123/aim3_unit_trays_results.csv"
chunk_size = 100000  # Adjust based on available memory
chunks = pd.read_csv(path, chunksize=chunk_size)
df2 = pd.concat(chunks, ignore_index=True)
print(f"Final DataFrame shape of Unit Tray CSV: {df2.shape}")

print("Number of rehoused drawers: ", len(df1[df1['is_rehoused_boolean'] == True]))

drawer_used_rate = sum(df1['total_area']) / sum(df1['drawer_box_area']) * 100
print("Drawer used rate: ", drawer_used_rate)

notes_used_rate = sum(df1['note_area_total']) / sum(df1['drawer_box_area']) * 100
print("Notes used rate: ", notes_used_rate)

# Number of full unit trays before expansion
df2_filtered = df2[df2['is_full'] == True]
print("Number of full unit trays: ", len(df2_filtered))

# Compute the expansion rate difference
# Compute the expansion rate 1
ratio1 = df1['aim2_expansion_percentage'].dropna().mean()
print("The mean expansion percentage calculated by averaging the drawer area differences.: ", ratio1)

# Filter rows where has_barcode is True
df_filtered = df[df['has_barcode'] == True]
# Compute the expansion rate 2
ratio2 = (df_filtered['specimen_area_with_barcode'].sum() - df_filtered['specimen_area'].sum()) / df_filtered['specimen_area'].sum()
print("Average expansion rate difference computed across barcode-labeled specimens.: ", ratio2)

# Create a mask for rows where is_unit_tray is False
mask = df1['is_unit_tray'] == False
full_ratio = 0.7

# Compute the new columns conditionally
df1.loc[mask, 'aim4_expansion_area_1'] = df1.loc[mask, 'aim2_unpaired_occupancy_area'] * ratio1
df1.loc[mask, 'aim4_expansion_area_2'] = df1.loc[mask, 'aim2_unpaired_occupancy_area'] * ratio2

df1.loc[mask, 'need_new_drawer_1'] = (df1.loc[mask, 'total_area'] + df1.loc[mask, 'aim4_expansion_area_1']) > (df1.loc[mask, 'drawer_box_area'] * full_ratio)
df1.loc[mask, 'need_new_drawer_2'] = (df1.loc[mask, 'total_area'] + df1.loc[mask, 'aim4_expansion_area_2']) > (df1.loc[mask, 'drawer_box_area'] * full_ratio)

print("Check the number of drawers required for non-unit-tray drawers after expansion:")
print(df1['need_new_drawer_1'].value_counts())
print()
print(df1['need_new_drawer_2'].value_counts())

# Add new columns based on the expansion ratios for the unit_tray csv
df2['aim4_expansion_area_1'] = df2['non_overlapping_specimen_area_12'] * ( 1 + ratio1 )
df2['aim4_expansion_area_2'] = df2['non_overlapping_specimen_area_12'] * ( 1 + ratio2 )

df2['need_new_tray_1'] = (df2['non_overlapping_specimen_area_11'] + df2['aim4_expansion_area_1']) > (df2['unit_tray_area'] * 0.8)
df2['need_new_tray_2'] = (df2['non_overlapping_specimen_area_11'] + df2['aim4_expansion_area_2']) > (df2['unit_tray_area'] * 0.8)

# Columns to copy
columns_to_copy = [
    'filename',
    'class_label_number',
    'normalized_center_x',
    'normalized_center_y',
    'normalized_top_left_x',
    'normalized_top_left_y',
    'normalized_bottom_right_x',
    'normalized_bottom_right_y',
    'unit_tray_area'
]
# Filter and create the new DataFrames
df_tray1 = df2[df2['need_new_tray_1'] == True][columns_to_copy].copy()
df_tray2 = df2[df2['need_new_tray_2'] == True][columns_to_copy].copy()

average_drawer_area = df1['drawer_box_area'].mean()
print("Average drawer area: ", average_drawer_area)

extra_drawer_needed1 = sum(df_tray1['unit_tray_area']) / average_drawer_area
extra_drawer_needed2 = sum(df_tray2['unit_tray_area']) / average_drawer_area

print("Extra drawer needed for tray 1: ", extra_drawer_needed1)
print("Extra drawer needed for tray 2: ", extra_drawer_needed2)