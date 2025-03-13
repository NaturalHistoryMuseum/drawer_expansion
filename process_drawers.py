import cv2
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import warnings

# Ignore FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)

def compute_area(bbox):
    width = bbox['normalized_width']
    height = bbox['normalized_height']
    return width * height

def get_center(bbox):
    x_center = bbox['normalized_center_x']
    y_center = bbox['normalized_center_y']
    return (x_center, y_center)

def compute_center(bbox):
    """
    Get the center of a bounding box.
    if bbox: [x_min, y_min, x_max, y_max]
    """
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    return (x_center, y_center)

def compute_width_height(bbox):
    """
    Compute the width and height of a bounding box.
    bbox: [x_min, y_min, x_max, y_max] (normalized coordinates)
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height

def compute_area_box(bbox, image_width, image_height):
    x_min = int(bbox['normalized_top_left_x'] * image_width)
    y_min = int(bbox['normalized_top_left_y'] * image_height)
    x_max = int(bbox['normalized_bottom_right_x'] * image_width)
    y_max = int(bbox['normalized_bottom_right_y'] * image_height)
    return (x_max - x_min) * (y_max - y_min)

def is_unit_tray_drawer(bboxes):
    if (5 in bboxes['class_label_number'].values):
        return True
    else:
        return False

def is_rehoused(bboxes):
    if len(bboxes[bboxes['class_label_number'] == 1])/len(bboxes[bboxes['class_label_number'] == 2]) > 0.8:
        return True
    else:
        return False

def euclidean_distance(point1, point2):
    """
    Compute the Euclidean distance between two points.
    point1: (x1, y1)
    point2: (x2, y2)
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def create_larger_bbox(specimen_bbox, label_bbox):
    """
    Create a larger bounding box that covers both the specimen and its nearest label.
    specimen_bbox: [x_min, y_min, x_max, y_max]
    label_bbox: [x_min, y_min, x_max, y_max]
    """
    x_left = min(specimen_bbox['normalized_top_left_x'], label_bbox['normalized_top_left_x'])
    y_left = min(specimen_bbox['normalized_top_left_y'], label_bbox['normalized_top_left_y'])
    x_right = max(specimen_bbox['normalized_bottom_right_x'], label_bbox['normalized_bottom_right_x'])
    y_right = max(specimen_bbox['normalized_bottom_right_y'], label_bbox['normalized_bottom_right_y'])
    
    return [x_left, y_left, x_right, y_right]

def find_nearest_label(specimen_bbox, label_bboxes):
    """
    Find the nearest label for a given specimen.
    specimen_bbox: [x_min, y_min, x_max, y_max]
    label_bboxes: List of label bounding boxes [[x_min, y_min, x_max, y_max], ...]
    """
    specimen_center = get_center(specimen_bbox)
    min_distance = float('inf')
    nearest_label_bbox = None
    nearest_label_bbox_index = None
    
    for index, label_bbox in label_bboxes.iterrows():
        label_center = get_center(label_bbox)
        # print(f"Label center: {label_center}")
        distance = euclidean_distance(specimen_center, label_center)
        # print(f"Distance: {distance}")
        if distance < min_distance:
            min_distance = distance
            nearest_label_bbox = label_bbox
            nearest_label_bbox_index = index
            # print(f"Nearest label bbox: {nearest_label_bbox}")
    
    return nearest_label_bbox, nearest_label_bbox_index, min_distance

def process_specimens_and_labels(specimen_bboxes, label_bboxes):
    """
    Process specimens and their nearest labels while ensuring that only the required fields are kept.
    If multiple specimens share the same nearest label, only the closest one is paired.
    Returns paired specimens, unpaired specimens, and unpaired labels with filtered fields.

    specimen_bboxes: DataFrame with specimen bounding boxes
    label_bboxes: DataFrame with label bounding boxes

    Returns:
    - paired_results: List of dictionaries containing paired specimens and labels.
    - unpaired_specimens: List of dictionaries with unpaired specimens.
    - unpaired_labels: List of dictionaries with unpaired labels.
    """
    fieldnames = [
        'class_label_number', 'normalized_center_x', 'normalized_center_y', 'normalized_width', 'normalized_height',
        'normalized_top_left_x', 'normalized_top_left_y', 'normalized_bottom_right_x', 'normalized_bottom_right_y'
    ]

    paired_results = []
    label_to_best_specimen = {}  # Stores the best specimen index for each label
    assigned_specimens = set()
    assigned_labels = set()

    specimen_data = []
    all_label_indices = set(label_bboxes.index)  # Track all label indices

    # First, determine the nearest label for each specimen
    for index, specimen_bbox in specimen_bboxes.iterrows():
        specimen_area = compute_area(specimen_bbox)
        
        # Get the nearest label and its distance
        nearest_label_bbox, nearest_label_bbox_index, distance = find_nearest_label(specimen_bbox, label_bboxes)

        if not nearest_label_bbox.empty:
            specimen_data.append({
                'specimen_index': index,
                'specimen_bbox': specimen_bbox[fieldnames],  # Keep only required fields
                'specimen_area': specimen_area,
                'nearest_label_bbox': nearest_label_bbox[fieldnames],  # Keep only required fields
                'nearest_label_bbox_index': nearest_label_bbox_index,
                'distance': distance
            })

    # Process specimens and filter to keep only the closest ones
    for data in specimen_data:
        label_index = data['nearest_label_bbox_index']

        # If this label is not assigned yet or this specimen is closer, update the assignment
        if label_index not in label_to_best_specimen or data['distance'] < label_to_best_specimen[label_index]['distance']:
            label_to_best_specimen[label_index] = data

    # Store only the best matches and track assigned specimens & labels
    for label_index, best_specimen in label_to_best_specimen.items():
        larger_bbox = create_larger_bbox(best_specimen['specimen_bbox'], best_specimen['nearest_label_bbox'])

        paired_results.append({
            'specimen_bbox': best_specimen['specimen_bbox'].to_dict(),
            'specimen_area': best_specimen['specimen_area'],
            'nearest_label_bbox': best_specimen['nearest_label_bbox'].to_dict(),
            'label_area': compute_area(best_specimen['nearest_label_bbox']),
            'larger_bbox': larger_bbox
        })

        assigned_specimens.add(best_specimen['specimen_index'])
        assigned_labels.add(label_index)

    # Find unpaired specimens (those not in assigned_specimens)
    unpaired_specimens = [
        specimen_bboxes.loc[i, fieldnames].to_dict() for i in specimen_bboxes.index if i not in assigned_specimens
    ]

    # Find unpaired labels (those not in assigned_labels)
    unpaired_labels = [
        label_bboxes.loc[i, fieldnames].to_dict() for i in label_bboxes.index if i not in assigned_labels
    ]

    return paired_results, unpaired_specimens, unpaired_labels

def convert_to_dataframe(bounding_boxes, label_value):
    """
    Convert a list of bounding boxes into a DataFrame with the specified fields.
    """
    fieldnames = [
        'class_label_number', 'normalized_center_x', 'normalized_center_y',
        'normalized_width', 'normalized_height',
        'normalized_top_left_x', 'normalized_top_left_y',
        'normalized_bottom_right_x', 'normalized_bottom_right_y'
    ]

    data = []
    for bbox in bounding_boxes:
        x_center, y_center = compute_center(bbox)
        width, height = compute_width_height(bbox)

        data.append({
            'class_label_number': label_value,  # Placeholder, update if needed
            'normalized_center_x': x_center,
            'normalized_center_y': y_center,
            'normalized_width': width,
            'normalized_height': height,
            'normalized_top_left_x': bbox[0],
            'normalized_top_left_y': bbox[1],
            'normalized_bottom_right_x': bbox[2],
            'normalized_bottom_right_y': bbox[3]
        })

    return pd.DataFrame(data, columns=fieldnames)

def filter_bboxes_inside_drawer(merged_df, drawer_bboxes):
    """
    Remove all bounding boxes in merged_df that are outside of the drawer box.

    Parameters:
    - merged_df: DataFrame containing bounding boxes.
    - drawer_bboxes: List of drawer bounding boxes (in normalized coordinates).

    Returns:
    - Filtered DataFrame with only bounding boxes inside the drawer box.
    """
    if drawer_bboxes.empty:
        return merged_df  # No drawer, return original DataFrame

    # Extract the first drawer box (assuming only one drawer is relevant)
    drawer_bbox = drawer_bboxes.iloc[0]  

    # Get drawer boundaries (normalized coordinates)
    drawer_x_min = drawer_bbox['normalized_top_left_x']
    drawer_y_min = drawer_bbox['normalized_top_left_y']
    drawer_x_max = drawer_bbox['normalized_bottom_right_x']
    drawer_y_max = drawer_bbox['normalized_bottom_right_y']

    # Filter: Keep only bounding boxes that are completely inside the drawer
    filtered_df = merged_df[
        (merged_df['normalized_top_left_x'] >= drawer_x_min) &  # Top-left X inside
        (merged_df['normalized_top_left_y'] >= drawer_y_min) &  # Top-left Y inside
        (merged_df['normalized_bottom_right_x'] <= drawer_x_max) &  # Bottom-right X inside
        (merged_df['normalized_bottom_right_y'] <= drawer_y_max)  # Bottom-right Y inside
    ]

    return filtered_df

def compute_total_non_overlapping_area(df, image_width, image_height):
    """
    Compute the total area of bounding boxes, ensuring overlapped regions are counted only once.

    Parameters:
    - df: Pandas DataFrame with bounding boxes.
    - image_width: Width of the image (to convert normalized coordinates).
    - image_height: Height of the image (to convert normalized coordinates).

    Returns:
    - Total non-overlapping area of bounding boxes.
    """
    polygons = []

    for _, row in df.iterrows():
        x_min = row['normalized_top_left_x'] * image_width
        y_min = row['normalized_top_left_y'] * image_height
        x_max = row['normalized_bottom_right_x'] * image_width
        y_max = row['normalized_bottom_right_y'] * image_height

        # Create a bounding box polygon
        bbox_polygon = box(x_min, y_min, x_max, y_max)
        polygons.append(bbox_polygon)

    # Compute the union of all bounding boxes to remove overlapping areas
    merged_polygon = unary_union(polygons)

    # Get the total non-overlapping area
    total_area = merged_polygon.area

    return total_area

def draw_bounding_boxes(image, df, total_area, image_width, image_height):
    """
    Draw bounding boxes on the image and display the total non-overlapping area.

    Parameters:
    - image: The input image.
    - df: DataFrame containing bounding boxes.
    - total_area: The computed total non-overlapping area.
    - image_width: Image width for denormalization.
    - image_height: Image height for denormalization.

    Returns:
    - The image with drawn bounding boxes and total area displayed.
    """
    # Make a copy of the image to avoid modifying the original
    image_with_boxes = image.copy()

    for _, row in df.iterrows():
        # Convert normalized coordinates to pixel values
        x_min = int(row['normalized_top_left_x'] * image_width)
        y_min = int(row['normalized_top_left_y'] * image_height)
        x_max = int(row['normalized_bottom_right_x'] * image_width)
        y_max = int(row['normalized_bottom_right_y'] * image_height)

        # Draw the bounding box (green)
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the total area on the image
    text = f"Total Non-Overlapping Area: {total_area:.2f} pixels"
    cv2.putText(image_with_boxes, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image_with_boxes

def visualize_merged_polygon(df, image_width, image_height):
    """
    Visualizes the merged bounding box polygon without displaying the original image.
    
    Parameters:
    - df: Pandas DataFrame containing bounding box information.
    - image_width: Width of the image (for coordinate scaling).
    - image_height: Height of the image (for coordinate scaling).
    """
    polygons = []

    # Convert bounding boxes to Shapely polygons
    for _, row in df.iterrows():
        x_min = row['normalized_top_left_x'] * image_width
        y_min = row['normalized_top_left_y'] * image_height
        x_max = row['normalized_bottom_right_x'] * image_width
        y_max = row['normalized_bottom_right_y'] * image_height

        bbox_polygon = box(x_min, y_min, x_max, y_max)
        polygons.append(bbox_polygon)

    # Merge overlapping bounding boxes
    merged_polygon = unary_union(polygons)

    # Create a plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the individual bounding boxes
    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.plot(x, y, color='gray', linestyle='--', linewidth=1)

    # Plot the merged bounding region
    if merged_polygon.geom_type == 'Polygon':
        x, y = merged_polygon.exterior.xy
        ax.fill(x, y, color='red', alpha=0.5)
        ax.plot(x, y, color='black', linewidth=2)
    else:  # MultiPolygon case
        for poly in merged_polygon.geoms:
            x, y = poly.exterior.xy
            ax.fill(x, y, color='red', alpha=0.5)
            ax.plot(x, y, color='black', linewidth=2)

    # Invert Y-axis to match image coordinates
    ax.set_ylim(image_height, 0)  # Set origin at the top-left like in images

    # Remove unnecessary labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)  # Remove plot borders

    # Show the plot
    plt.show()

def find_nearest_specimen_for_barcodes(specimen_bboxes, barcode_bboxes):
    """
    Ensure that each barcode is assigned to the closest available specimen.
    Not all specimens need to be assigned.

    Parameters:
    - specimen_bboxes: DataFrame containing specimen bounding boxes.
    - barcode_bboxes: DataFrame containing barcode bounding boxes.

    Returns:
    - Dictionary mapping barcode index -> nearest specimen index.
    """
    if specimen_bboxes.empty or barcode_bboxes.empty:
        return {}  # Return empty dictionary if no data

    assignments = {}  # Maps barcode index to specimen index
    specimen_assigned = set()  # Track assigned specimens
    distance_list = []

    # Compute distances between all barcode boxes and specimens
    for b_idx, barcode_bbox in barcode_bboxes.iterrows():
        barcode_center = get_center(barcode_bbox)

        for s_idx, specimen_bbox in specimen_bboxes.iterrows():
            specimen_center = get_center(specimen_bbox)
            distance = euclidean_distance(barcode_center, specimen_center)
            distance_list.append((distance, b_idx, s_idx))

    # Sort distances in ascending order
    distance_list.sort()

    # Assign barcodes to the closest specimen based on shortest distance
    for distance, b_idx, s_idx in distance_list:
        if b_idx not in assignments and s_idx not in specimen_assigned:
            assignments[b_idx] = s_idx
            specimen_assigned.add(s_idx)  # Mark this specimen as used

    return assignments  # Maps barcode index -> nearest specimen index

def process_specimens_and_barcodes(specimen_bboxes, barcode_bboxes):
    """
    Process all specimens and assign each a unique nearest barcode.

    Parameters:
    - specimen_bboxes: DataFrame of specimen bounding boxes.
    - barcode_bboxes: DataFrame of barcode bounding boxes.

    Returns:
    - List of dictionaries with specimen, barcode, and bounding box information.
    """
    if specimen_bboxes.empty or barcode_bboxes.empty:
        return []  # Return empty list if no valid data

    results = []
    barcode_assignments = find_nearest_specimen_for_barcodes(specimen_bboxes, barcode_bboxes)

    for b_idx, s_idx in barcode_assignments.items():
        specimen_bbox = specimen_bboxes.loc[s_idx]
        nearest_barcode_bbox = barcode_bboxes.loc[b_idx]

        # Compute specimen width & height
        specimen_width = specimen_bbox['normalized_bottom_right_x'] - specimen_bbox['normalized_top_left_x']
        specimen_height = specimen_bbox['normalized_bottom_right_y'] - specimen_bbox['normalized_top_left_y']

        # Create the larger bounding box
        larger_bbox = create_larger_bbox(specimen_bbox, nearest_barcode_bbox)

        # Compute larger box width & height
        larger_width, larger_height = compute_width_height(larger_bbox)

        # Check if the larger bounding box exceeds 1.5x the specimen size
        if larger_width > 1.5 * specimen_width or larger_height > 1.5 * specimen_height:
            continue  # Skip this match

        # Compute areas for reference
        specimen_area = compute_area(specimen_bbox)
        barcode_area = compute_area(nearest_barcode_bbox)  

        results.append({
            'specimen_bbox': specimen_bbox.to_dict(),  # Ensure dictionary format
            'specimen_area': specimen_area,
            'nearest_barcode_bbox': nearest_barcode_bbox.to_dict(),
            'barcode_area': barcode_area,
            'larger_bbox': larger_bbox
        })

    return results

def analyze_unit_tray_occupancy(merged_df, unit_tray_bboxes, image_width, image_height, full_threshold=0.8):
    """
    Analyze each unit tray's occupancy:
    1. Filter bounding boxes inside the unit tray.
    2. Calculate the non-overlapping specimen area.
    3. Determine whether the unit tray is full.

    Parameters:
    - merged_df: DataFrame of all bounding boxes.
    - unit_tray_bboxes: DataFrame of unit tray bounding boxes.
    - image_width: Image width (for coordinate conversion).
    - image_height: Image height (for coordinate conversion).
    - full_threshold: Percentage threshold to consider the tray as full (default=80%).

    Returns:
    - DataFrame with unit tray ID, non-overlapping area, and occupancy status.
    """
    results = []

    for idx in range(len(unit_tray_bboxes)):
        tray_bbox = unit_tray_bboxes.iloc[idx]  # Get tray bounding box

        # Filter bounding boxes inside this unit tray
        filtered_bboxes = filter_bboxes_inside_drawer(merged_df, unit_tray_bboxes.iloc[idx:idx+1])

        # Compute non-overlapping specimen area in this tray
        non_overlapping_area = compute_total_non_overlapping_area(filtered_bboxes, image_width, image_height)

        # Compute unit tray area
        tray_area = compute_area_box(tray_bbox, image_width, image_height)

        # Determine if the unit tray is full
        is_full = (non_overlapping_area / tray_area) >= full_threshold

        # Store results
        results.append({
            'unit_tray_index': idx,
            'non_overlapping_specimen_area': non_overlapping_area,
            'unit_tray_area': tray_area,
            'occupancy_ratio': non_overlapping_area / tray_area,
            'is_full': is_full
        })

    return pd.DataFrame(results)

# Define function to process each CSV
def process_drawer_data(csv_path, image_dir, output_dir_aim2, output_dir_aim3):
    """
    Process a single CSV file and generate AIM2 and AIM3 results.

    Parameters:
    - csv_path: Path to the CSV file.
    - image_dir: Directory containing images.
    - output_dir_aim2: Directory to save AIM2 results.
    - output_dir_aim3: Directory to save AIM3 results.
    """

    # Extract filename (without extension) to find corresponding image
    filename = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Construct image path
    image_path = os.path.join(image_dir, f"{filename}.jpeg")  # Adjust extension if needed

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return

    # Get image dimensions
    image_height, image_width, _ = image.shape

    # Load bounding box data
    bboxes = pd.read_csv(csv_path)

    # Extract different class bounding boxes
    specimen_bboxes = bboxes[bboxes['class_label_number'] == 0]
    barcode_bboxes = bboxes[bboxes['class_label_number'] == 1]
    label_bboxes = bboxes[bboxes['class_label_number'] == 2]
    note_bboxes = bboxes[bboxes['class_label_number'] == 3]
    drawer_bboxes = bboxes[bboxes['class_label_number'] == 4]
    unit_tray_bboxes = bboxes[bboxes['class_label_number'] == 5]

    # Aim 1: Current occupancy estimate
    results, unpaired_specimens, unpaired_labels = process_specimens_and_labels(specimen_bboxes, label_bboxes)
    bounding_boxes = [result['larger_bbox'] for result in results]
    bounding_boxes_df = convert_to_dataframe(bounding_boxes, 0)
    
    # Convert unpaired to DataFrames
    unpaired_specimens_df = pd.DataFrame(unpaired_specimens)
    unpaired_labels_df = pd.DataFrame(unpaired_labels)
    
    # Merge bounding boxes
    merged_df = pd.concat([bounding_boxes_df, unpaired_specimens_df, unpaired_labels_df, barcode_bboxes, note_bboxes], ignore_index=True)
    
    # Filter by drawer
    occupancy_df = filter_bboxes_inside_drawer(merged_df, drawer_bboxes)

    # Compute total non-overlapping area
    total_area = compute_total_non_overlapping_area(occupancy_df, image_width, image_height)

    # Aim 2: Specimen expansion with barcode
    occupancy_df.loc[occupancy_df['class_label_number'].isin([0, 2]), 'class_label_number'] = 10
    occupancy_bboxes = occupancy_df[occupancy_df['class_label_number'] == 10]
    
    aim2_results = process_specimens_and_barcodes(occupancy_bboxes, barcode_bboxes)
    aim2_bounding_boxes = [result['larger_bbox'] for result in aim2_results]
    aim2_specimen_bboxes = [result['specimen_bbox'] for result in aim2_results]
    aim2_specimen_bboxes_df = pd.DataFrame(aim2_specimen_bboxes)
    aim2_bounding_boxes_df = convert_to_dataframe(aim2_bounding_boxes, 11)
    
    # Compute areas
    aim2_A0 = compute_total_non_overlapping_area(aim2_bounding_boxes_df, image_width, image_height)
    aim2_A = compute_total_non_overlapping_area(aim2_specimen_bboxes_df, image_width, image_height)

    # Merge Aim 2 results
    aim2_df = pd.concat([aim2_specimen_bboxes_df, aim2_bounding_boxes_df], ignore_index=True)

    # Save Aim 2 results
    aim2_output_path = os.path.join(output_dir_aim2, f"{filename}.csv")
    aim2_df.to_csv(aim2_output_path, index=False)
    print(f"AIM2 saved: {aim2_output_path}")

    # Aim 3: Calculate mean occupancy of unit trays
    if is_unit_tray_drawer(occupancy_df):
        unit_tray_occupancy_df = analyze_unit_tray_occupancy(occupancy_df, unit_tray_bboxes, image_width, image_height)
    
        # Merge unit tray occupancy info
        unit_tray_bboxes = unit_tray_bboxes.reset_index()
        merged_unit_trays_df = unit_tray_bboxes.merge(
            unit_tray_occupancy_df, 
            left_index=True, 
            right_on="unit_tray_index", 
            how="left"
        ).drop(columns=["unit_tray_index"])
    
        merged_unit_trays_df = merged_unit_trays_df.set_index("index")
    
        # Merge all Aim 3 data
        aim3_df = pd.concat([occupancy_df, drawer_bboxes, merged_unit_trays_df], ignore_index=True)

    # Save Aim 3 results
    aim3_output_path = os.path.join(output_dir_aim3, f"{filename}.csv")
    aim3_df.to_csv(aim3_output_path, index=False)
    print(f"AIM3 saved: {aim3_output_path}")

# Main function to process all CSVs in a directory
def process_all_drawers(image_dir, csv_dir, output_dir_aim2, output_dir_aim3):
    """
    Process all CSV files in the given directory.

    Parameters:
    - image_dir: Directory containing images.
    - csv_dir: Directory containing CSV files.
    - output_dir_aim2: Directory to save AIM2 results.
    - output_dir_aim3: Directory to save AIM3 results.
    """
    # Ensure output directories exist
    os.makedirs(output_dir_aim2, exist_ok=True)
    os.makedirs(output_dir_aim3, exist_ok=True)

    # Loop through CSV files in the directory
    for csv_filename in os.listdir(csv_dir):
        if csv_filename.endswith(".csv"):  # Process only CSV files
            csv_path = os.path.join(csv_dir, csv_filename)
            process_drawer_data(csv_path, image_dir, output_dir_aim2, output_dir_aim3)

# Example usage (Modify with actual paths)
image_directory = "/Users/mbax9qg2/Downloads/training_data/source/"
csv_directory = "/Users/mbax9qg2/Downloads/processed/"
output_directory_aim2 = "/Users/mbax9qg2/Downloads/processed/aim2/"
output_directory_aim3 = "/Users/mbax9qg2/Downloads/processed/aim3/"

# Run the script
process_all_drawers(image_directory, csv_directory, output_directory_aim2, output_directory_aim3)
