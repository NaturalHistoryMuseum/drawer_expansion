import os
import pandas as pd

def is_unit_tray_drawer(bboxes):
    return 5 in bboxes['class_label_number'].values

def is_drawer(bboxes):
    return 4 in bboxes['class_label_number'].values

def process_csv_files(directory):
    total_tray_drawer = 0
    total_drawer = 0
    
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            csv_path = os.path.join(directory, file)
            bboxes = pd.read_csv(csv_path)
            
            if is_unit_tray_drawer(bboxes):
                total_tray_drawer += 1
            if is_drawer(bboxes):
                total_drawer += 1
    
    return total_tray_drawer, total_drawer

# Example usage
directory_path = "C:/Users/qiang/Documents/coding/drawer/processed/merged_with_barcode"  # Change this to your directory
tray_drawer_count, drawer_count = process_csv_files(directory_path)
print(f"Total is_unit_tray_drawer: {tray_drawer_count}") # 8757
print(f"Total is_drawer: {drawer_count}") # 11045