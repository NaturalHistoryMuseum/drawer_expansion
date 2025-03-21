import os
import pandas as pd

def is_unit_tray_drawer(bboxes):
    return 5 in bboxes['class_label_number'].values

def is_drawer(bboxes):
    return 4 in bboxes['class_label_number'].values

def process_csv_files(directory):
    total_tray_drawer = 0
    total_drawer = 0
    non_drawer_files = []
    drawer_data = []
    
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            csv_path = os.path.join(directory, file)
            bboxes = pd.read_csv(csv_path)
            
            if is_unit_tray_drawer(bboxes):
                total_tray_drawer += 1
            
            if is_drawer(bboxes):
                total_drawer += 1
                drawer_data.append(bboxes[bboxes['class_label_number'] == 4])
            else:
                non_drawer_files.append(file)
    
    # Save non-drawer file names as CSV
    non_drawer_df = pd.DataFrame({'filename': non_drawer_files})
    non_drawer_csv_path = os.path.join(directory, "non_drawer_files.csv")
    non_drawer_df.to_csv(non_drawer_csv_path, index=False)
    
    # Concatenate all drawer data and compute averages
    if drawer_data:
        all_drawers = pd.concat(drawer_data, ignore_index=True)
        drawer_averages = all_drawers[['normalized_center_x', 'normalized_center_y', 'normalized_width', 'normalized_height',
                                       'normalized_top_left_x', 'normalized_top_left_y', 'normalized_bottom_right_x', 'normalized_bottom_right_y']].mean()
        drawer_avg_df = pd.DataFrame([drawer_averages])
        drawer_avg_csv_path = os.path.join(directory, "drawer_averages.csv")
        drawer_avg_df.to_csv(drawer_avg_csv_path, index=False)
    else:
        drawer_averages = {}
    
    return total_tray_drawer, total_drawer, drawer_averages

# Example usage
directory_path = "C:/Users/qiang/Documents/coding/drawer/processed/merged_with_barcode"  # Change this to your directory
tray_drawer_count, drawer_count, drawer_avg = process_csv_files(directory_path)
print(f"Total is_unit_tray_drawer: {tray_drawer_count}")  # 8757
print(f"Total is_drawer: {drawer_count}")  # 11045
print(f"Drawer Averages: {drawer_avg}")