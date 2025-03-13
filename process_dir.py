import os
import cv2
from ultralytics import YOLO
import csv
from tqdm import tqdm 

# Function to save results in the desired format with class label numbers
def save_results_to_csv(results, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow([
            'class_label_number', 'normalized_center_x', 'normalized_center_y', 'normalized_width', 'normalized_height',
            'normalized_top_left_x', 'normalized_top_left_y', 'normalized_bottom_right_x', 'normalized_bottom_right_y',
            'center_x', 'center_y', 'width', 'height', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence'
        ])
        
        # Write each detection result
        for result in results:
            # result.show()
            xywhn = result.boxes.xywhn  # Normalized center-x, center-y, width, height
            xyxyn = result.boxes.xyxyn  # Normalized top-left-x, top-left-y, bottom-right-x, bottom-right-y
            class_indices = result.boxes.cls.int().tolist()  # Class label numbers
            xywh = result.boxes.xywh  # center-x, center-y, width, height
            xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
            confs = result.boxes.conf  # Confidence scores
            
            for i in range(len(class_indices)):
                class_label_number = class_indices[i]  # Class label number
                center_x, center_y, width, height = xywhn[i].tolist()
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = xyxyn[i].tolist()
                x, y, w, h = xywh[i].tolist()
                x_min, y_min, x_max, y_max = xyxy[i].tolist()
                confidence = confs[i].item()
                
                # Write the row to the CSV file
                writer.writerow([
                    class_label_number, center_x, center_y, width, height,
                    top_left_x, top_left_y, bottom_right_x, bottom_right_y,
                    x, y, w, h, x_min, y_min, x_max, y_max, confidence
                ])

# Example function that processes an image
def process_image(image, model):
    # Replace this with your actual image processing logic
    # print("Processing image")
    results = model(image)
    return results

# Function to process all images in a directory
def process_images_in_directory(directory_path, output_directory, process_function, model):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Get the list of image files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    print(f"Total images to process: {total_images}")

    # Initialize tqdm progress bar
    for filename in tqdm(image_files, desc="Processing images", unit="image"):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)

        # Check if the file is an image (you can add more extensions if needed)
        try:
            # Read the image using OpenCV
            img = cv2.imread(file_path)
            if img is None:
                print(f"Failed to load image: {filename}")
                continue

            # Process the image
            processed_image_results = process_function(img, model)

            # Save the results as a CSV file with the same name as the image
            output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.csv")
            save_results_to_csv(processed_image_results, output_path)
            print(f"Processed image saved to: {output_path}")

        except Exception as e:
            print(f"Error processing image {filename}: {e}")

# Example usage
directory_path = "/Users/mbax9qg2/Downloads/training_data/source"
output_directory = "/Users/mbax9qg2/Downloads/processed"
model = YOLO('/Users/mbax9qg2/Documents/drawer_dataset/train2/weights/best.pt')
process_images_in_directory(directory_path, output_directory, process_image, model)