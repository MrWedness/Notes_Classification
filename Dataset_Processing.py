import numpy as np
from PIL import ImageDraw, Image
import os

def process_dataset_Class_Filter(root_dir, canvas_height=28, canvas_width=28, padding=2, labels_to_keep=None, stroke_thickness=3):

    def parse_txt_file(txt_file_path):
        """We grab the strokes from the txt files"""
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()

        label = lines[0].strip()  # The first line contains the label
        strokes = []

        for line in lines[1:]:  # Remaining lines are strokes
            stroke_points = []
            points = line.strip().split(';')  # Each point is separated by a semicolon
            for point in points:
                if not point:
                    continue  # Skip empty points
                try:
                    x, y = map(int, point.split(','))  # Convert each point to integers
                    stroke_points.append((x, y))
                except ValueError:
                    print(f"Warning: Skipping invalid point '{point}' in line '{line.strip()}'")
            strokes.append(stroke_points)

        return label, strokes

    def center_and_scale(points, canvas_height, canvas_width, padding):
        """This centres the points on the canvas"""
        points = np.array(points, dtype=np.float64)  # Ensure points are in numpy array format and of type float64

        # Calculate bounding box
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)

        # Translate points to center them
        centroid_x = (min_x + max_x) / 2
        centroid_y = (min_y + max_y) / 2
        points[:, 0] -= centroid_x
        points[:, 1] -= centroid_y

        # Scale points to fit within the canvas
        max_dim = max(max_x - min_x, max_y - min_y)
        scale_x = (canvas_width - 2 * padding) / max_dim
        scale_y = (canvas_height - 2 * padding) / max_dim
        scale = min(scale_x, scale_y)  # Use the smaller scale to preserve aspect ratio
        points *= scale

        # Shift points to fit within the canvas with padding
        points[:, 0] += canvas_width / 2
        points[:, 1] += canvas_height / 2

        return points

    def rasterize(points, canvas_height, canvas_width, stroke_thickness):
        """Converts points to a rasterized image with specified stroke thickness."""
        # Create a blank white image
        image = Image.new("L", (canvas_width, canvas_height), 255)
        draw = ImageDraw.Draw(image)

        # Draw lines between consecutive points with the specified stroke thickness
        for i in range(len(points) - 1):
          # Convert points[i] and points[i + 1] to tuples
            draw.line([tuple(points[i]), tuple(points[i + 1])], fill=0, width=stroke_thickness)

      # Convert to numpy array
        return np.array(image)

    def create_image_from_strokes(strokes, canvas_height, canvas_width, padding, stroke_thickness):
        """Creates a rasterized image from strokes."""
        all_points = [point for stroke in strokes for point in stroke]  # Flatten all strokes into a single list
        centered_points = center_and_scale(all_points, canvas_height, canvas_width, padding)  # Center and scale points
        return rasterize(centered_points, canvas_height, canvas_width, stroke_thickness)  # Rasterize into an image

    # Lists to store the images and labels
    images = []
    labels = []

    # Process each folder (which represents a label)
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        if os.path.isdir(folder_path):  # If it's a folder
            for txt_file_name in os.listdir(folder_path):
                if txt_file_name.endswith('.txt'):
                    txt_file_path = os.path.join(folder_path, txt_file_name)

                    # Parse the txt file
                    label, strokes = parse_txt_file(txt_file_path)

                    # Filter based on specified labels
                    if labels_to_keep is not None and label not in labels_to_keep:
                        continue  # Skip if the label is not in the list to keep

                    # Generate the image from strokes
                    image = create_image_from_strokes(strokes, canvas_height, canvas_width, padding, stroke_thickness)

                    # Append image and label
                    images.append(image)
                    labels.append(label)

    # Convert images to numpy array
    images = np.array(images)
    labels = np.array(labels)

    # Normalize images (scale to [0, 1])
    images = images.astype('float32') / 255.0

    # Map labels to numerical values
    unique_labels = np.unique(labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    labels_numerical = np.array([label_mapping[label] for label in labels])

    return images, labels_numerical, label_mapping
