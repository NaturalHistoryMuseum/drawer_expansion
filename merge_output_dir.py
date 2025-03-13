import os
from ultralytics import YOLO
import csv
from tqdm import tqdm

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Each box is represented as a dictionary with keys:
    - normalized_top_left_x
    - normalized_top_left_y
    - normalized_bottom_right_x
    - normalized_bottom_right_y
    """
    # Get the coordinates of the intersection rectangle
    x1 = max(box1['normalized_top_left_x'], box2['normalized_top_left_x'])
    y1 = max(box1['normalized_top_left_y'], box2['normalized_top_left_y'])
    x2 = min(box1['normalized_bottom_right_x'], box2['normalized_bottom_right_x'])
    y2 = min(box1['normalized_bottom_right_y'], box2['normalized_bottom_right_y'])
    
    # Compute the area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute the area of both bounding boxes
    box1_area = (box1['normalized_bottom_right_x'] - box1['normalized_top_left_x']) * \
                (box1['normalized_bottom_right_y'] - box1['normalized_top_left_y'])
    box2_area = (box2['normalized_bottom_right_x'] - box2['normalized_top_left_x']) * \
                (box2['normalized_bottom_right_y'] - box2['normalized_top_left_y'])
    
    # Compute the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def merge_boxes(box1, box2):
    """
    Merge two bounding boxes into one.
    Each box is represented as a dictionary with keys:
    - normalized_top_left_x
    - normalized_top_left_y
    - normalized_bottom_right_x
    - normalized_bottom_right_y
    """
    merged_box = {
        'normalized_top_left_x': min(box1['normalized_top_left_x'], box2['normalized_top_left_x']),
        'normalized_top_left_y': min(box1['normalized_top_left_y'], box2['normalized_top_left_y']),
        'normalized_bottom_right_x': max(box1['normalized_bottom_right_x'], box2['normalized_bottom_right_x']),
        'normalized_bottom_right_y': max(box1['normalized_bottom_right_y'], box2['normalized_bottom_right_y'])
    }
    return merged_box

def merge_csv_files(csv_file1, csv_file2, output_file, iou_threshold=0.5):
    """
    Merge two CSV files of bounding box detections for the same image.
    If bounding boxes with the same class label overlap, merge them.
    Otherwise, copy the non-overlapping boxes to the output CSV.
    """
    # Read the first CSV file
    with open(csv_file1, mode='r') as file:
        reader = csv.DictReader(file)
        csv1_data = [row for row in reader]
    
    # Read the second CSV file
    with open(csv_file2, mode='r') as file:
        reader = csv.DictReader(file)
        csv2_data = [row for row in reader]
    
    # Combine data from both CSVs
    combined_data = csv1_data + csv2_data
    
    # Process the combined data
    merged_data = []
    used_indices = set()  # Track indices of merged boxes
    
    for i in range(len(combined_data)):
        if i in used_indices:
            continue  # Skip already merged boxes
        
        row1 = combined_data[i]
        class_label1 = int(row1['class_label_number'])
        box1 = {
            'normalized_top_left_x': float(row1['normalized_top_left_x']),
            'normalized_top_left_y': float(row1['normalized_top_left_y']),
            'normalized_bottom_right_x': float(row1['normalized_bottom_right_x']),
            'normalized_bottom_right_y': float(row1['normalized_bottom_right_y'])
        }
        
        # Check for overlapping boxes with the same class label
        for j in range(i + 1, len(combined_data)):
            if j in used_indices:
                continue  # Skip already merged boxes
            
            row2 = combined_data[j]
            class_label2 = int(row2['class_label_number'])
            box2 = {
                'normalized_top_left_x': float(row2['normalized_top_left_x']),
                'normalized_top_left_y': float(row2['normalized_top_left_y']),
                'normalized_bottom_right_x': float(row2['normalized_bottom_right_x']),
                'normalized_bottom_right_y': float(row2['normalized_bottom_right_y'])
            }
            
            if class_label1 == class_label2:
                iou = calculate_iou(box1, box2)
                if iou > iou_threshold:
                    # Merge the boxes
                    box1 = merge_boxes(box1, box2)
                    used_indices.add(j)  # Mark the second box as merged
        
        # Add the merged or non-overlapping box to the result
        merged_box = {
            'class_label_number': class_label1,
            'normalized_center_x': (box1['normalized_top_left_x'] + box1['normalized_bottom_right_x']) / 2,
            'normalized_center_y': (box1['normalized_top_left_y'] + box1['normalized_bottom_right_y']) / 2,
            'normalized_width': box1['normalized_bottom_right_x'] - box1['normalized_top_left_x'],
            'normalized_height': box1['normalized_bottom_right_y'] - box1['normalized_top_left_y'],
            'normalized_top_left_x': box1['normalized_top_left_x'],
            'normalized_top_left_y': box1['normalized_top_left_y'],
            'normalized_bottom_right_x': box1['normalized_bottom_right_x'],
            'normalized_bottom_right_y': box1['normalized_bottom_right_y']
        }
        merged_data.append(merged_box)
    
    # Write the merged data to the output CSV
    with open(output_file, mode='w', newline='') as file:
        fieldnames = [
            'class_label_number', 'normalized_center_x', 'normalized_center_y', 'normalized_width', 'normalized_height',
            'normalized_top_left_x', 'normalized_top_left_y', 'normalized_bottom_right_x', 'normalized_bottom_right_y'
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # Write the header
        writer.writerows(merged_data)  # Write the merged data


# Function to process all images in a directory
def merge_output_directory(directory_path1, directory_path2, output_directory):
    # Check if the directory exists
    if not os.path.exists(directory_path1):
        print(f"Directory '{directory_path1}' does not exist.")
        return
    
    if not os.path.exists(directory_path2):
        print(f"Directory '{directory_path2}' does not exist.")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Get the list of image files in the directory 1
    csv_files = [f for f in os.listdir(directory_path1) if f.lower().endswith(('.csv'))]
    total_csvs = len(csv_files)
    print(f"Total CSV files to process: {total_csvs}")

    # Initialize tqdm progress bar
    for filename in tqdm(csv_files, desc="Processing CSV files", unit="file"):

        # Check if the file is an image (you can add more extensions if needed)
        try:
            # Construct the full file paths
            csv_file1 = os.path.join(directory_path1, filename)
            csv_file2 = os.path.join(directory_path2, filename)
            output_file = os.path.join(output_directory, filename)

            # Check if the corresponding CSV file exists in directory 2
            if not os.path.exists(csv_file2):
                print(f"Corresponding CSV file '{filename}' does not exist in '{directory_path2}'.")
                continue

            # Process the image and save the results as a CSV file with the same name as the image
            merge_csv_files(csv_file1, csv_file2, output_file)
            print(f"Processed image saved to: {output_file}")

        except Exception as e:
            print(f"Error processing image {filename}: {e}")

# Example usage
directory_path1 = '/Users/mbax9qg2/Downloads/processed/model1'
directory_path2 = '/Users/mbax9qg2/Downloads/processed/model2'
output_directory = '/Users/mbax9qg2/Downloads/processed/merged'
merge_output_directory(directory_path1, directory_path2, output_directory)
print(f"Merged results saved to {output_directory}")