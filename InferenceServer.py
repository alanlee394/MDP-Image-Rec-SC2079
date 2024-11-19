import os
import time
import shutil
import glob
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO  # Import the YOLOv8 model from ultralytics
import numpy as np
from PIL import Image
import io
import sys
import cv2

app = Flask(__name__)
CORS(app)

# Load YOLOv8 model with the new weights
# Replace the path with the correct path to your model weights
model = YOLO('/Users/alanlee/documents/sc2079/yoloV5/ajbm.pt')





# Define the new class names from your updated model

new_model_class_names = {
    0: '1',
    1: '2',
    2: '3',
    3: '4',
    4: '5',
    5: '6',
    6: '7',
    7: '8',
    8: '9',
    9: 'A',
    10: 'B',
    11: 'Bullseye',
    12: 'C',
    13: 'D',
    14: 'Down',
    15: 'E',
    16: 'F',
    17: 'G',
    18: 'H',
    19: 'Left',
    20: 'Right',
    21: 'S',
    22: 'Stop',
    23: 'T',
    24: 'U',
    25: 'Up',
    26: 'V',
    27: 'W',
    28: 'X',
    29: 'Y',
    30: 'Z',
    # Class ID 31 ('z-') can be ignored
    31: 'z-'
}

# Update the custom class name mapping to match the new class names
custom_class_name_mapping = {
    # Numbers
    '1': 'Number One',
    '2': 'Number Two',
    '3': 'Number Three',
    '4': 'Number Four',
    '5': 'Number Five',
    '6': 'Number Six',
    '7': 'Number Seven',
    '8': 'Number Eight',
    '9': 'Number Nine',

    # Alphabets
    'A': 'Alphabet A',
    'B': 'Alphabet B',
    'C': 'Alphabet C',
    'D': 'Alphabet D',
    'E': 'Alphabet E',
    'F': 'Alphabet F',
    'G': 'Alphabet G',
    'H': 'Alphabet H',
    'S': 'Alphabet S',
    'T': 'Alphabet T',
    'U': 'Alphabet U',
    'V': 'Alphabet V',
    'W': 'Alphabet W',
    'X': 'Alphabet X',
    'Y': 'Alphabet Y',
    'Z': 'Alphabet Z',

    # Arrows
    'Up': 'Up arrow',
    'Down': 'Down arrow',
    'Right': 'Right arrow',
    'Left': 'Left arrow',

    # Stop Sign
    'Stop': 'Stop'
    # Note: 'Bullseye' is not mapped, as per your instruction
}

# Update the class-to-image ID mapping
class_to_image_id = {
    # Numbers
    'Number One': 11,
    'Number Two': 12,
    'Number Three': 13,
    'Number Four': 14,
    'Number Five': 15,
    'Number Six': 16,
    'Number Seven': 17,
    'Number Eight': 18,
    'Number Nine': 19,

    # Alphabets
    'Alphabet A': 20,
    'Alphabet B': 21,
    'Alphabet C': 22,
    'Alphabet D': 23,
    'Alphabet E': 24,
    'Alphabet F': 25,
    'Alphabet G': 26,
    'Alphabet H': 27,
    'Alphabet S': 28,
    'Alphabet T': 29,
    'Alphabet U': 30,
    'Alphabet V': 31,
    'Alphabet W': 32,
    'Alphabet X': 33,
    'Alphabet Y': 34,
    'Alphabet Z': 35,

    # Arrows
    'Up arrow': 36,
    'Down arrow': 37,
    'Right arrow': 38,
    'Left arrow': 39,

    # Stop Sign
    'Stop': 40
}

def sharpen_image(image):
    """
    Sharpen the input image using a predefined kernel.
    
    Parameters:
        image (np.ndarray): The input image in BGR format.
    
    Returns:
        np.ndarray: The sharpened image.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def apply_clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the input image.
    
    Parameters:
        image (np.ndarray): The input image in BGR format.
    
    Returns:
        np.ndarray: The image after CLAHE application.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    clahe_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return clahe_image

def denoise_image(image):
    """
    Denoise the input image using Non-Local Means Denoising.

    Parameters:
        image (np.ndarray): The input image in BGR format.

    Returns:
        np.ndarray: The denoised image.
    """
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
    )
    return denoised

def contrast_stretch(image):
    """Apply contrast stretching to enhance the image."""
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = 150 + (image - min_val) * (255 - 150 / (max_val - min_val))
    return np.uint8(stretched)

def predict_image_week_9(image_path, model, obstacle_id=None):
    """
    Processes the image using the YOLOv8 model, applies denoising,
    and returns a JSON response with obstacle_id and image_id of the selected object.
    """

    import cv2
    import numpy as np

    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if img is None:
        return {"error": "Failed to load image"}

    height, width, _ = img.shape
    image_center_x = width / 2
    image_center_y = height / 2

    # Apply denoising
    img_denoised = denoise_image(img)
    
    # Apply histogram equalization
    img_denoised = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2YUV)  # Convert to YUV color space
    img_denoised[:, :, 0] = cv2.equalizeHist(img_denoised[:, :, 0])  # Equalize the Y channel
    img_denoised = cv2.cvtColor(img_denoised, cv2.COLOR_YUV2BGR)  # Convert back to BGR
    
    img_denoised = contrast_stretch(img_denoised)
    
    # Use the model to make predictions
    results = model(img_denoised, conf=0.20)  # Adjust the confidence threshold

    # Initialize variables to track the largest bounding box within and outside the center
    largest_area_within_center = 0
    best_box_within_center = None
    detected_class_name_within_center = None
    image_id_within_center = None

    # Fallback: largest bounding box overall (if nothing is found within the center)
    largest_area_overall = 0
    best_box_overall = None
    detected_class_name_overall = None
    image_id_overall = None

    # Define the maximum allowed distance from the center (80%)
    max_distance = np.sqrt((image_center_x) ** 2 + (image_center_y) ** 2)
    distance_threshold = max_distance * 0.7  # Adjust this value if needed

    # Check if any detections are found
    if results is None or len(results) == 0 or len(results[0].boxes) == 0:
        print("No detections were made.")
        return {"error": "No detections found"}

    # Iterate over each detection
    for result in results:
        for box in result.boxes:
            class_idx = int(box.cls[0].cpu().numpy())  # Class index

            # Ignore specific class (e.g., 'z-')
            if class_idx == 31:
                continue

            # Get the class name from the new model's class mapping
            model_class_name = new_model_class_names.get(class_idx)
            if model_class_name is None:
                print(f"Unknown class ID {class_idx}. Skipping.")
                continue

            # Map to the original class name
            original_class_name = custom_class_name_mapping.get(model_class_name, model_class_name)

            # Skip "Bullseye" class
            if original_class_name == "Bullseye":
                continue

            # Extract bounding box coordinates and calculate the area
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)

            # Calculate the center of the bounding box
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2

            # Compute the distance from the image center to the bounding box center
            distance = np.sqrt((bbox_center_x - image_center_x) ** 2 + (bbox_center_y - image_center_y) ** 2)

            # Check if the bounding box is within the center area
            if distance <= distance_threshold:
                if area > largest_area_within_center:
                    largest_area_within_center = area
                    best_box_within_center = box
                    detected_class_name_within_center = original_class_name
                    image_id_within_center = class_to_image_id.get(detected_class_name_within_center, None)

            # Track the largest bounding box overall
            if area > largest_area_overall:
                largest_area_overall = area
                best_box_overall = box
                detected_class_name_overall = original_class_name
                image_id_overall = class_to_image_id.get(detected_class_name_overall, None)

    # Use the largest box within the center if available; otherwise, use the largest overall
    if best_box_within_center:
        selected_box = best_box_within_center
        detected_class_name = detected_class_name_within_center
        image_id = image_id_within_center
    else:
        selected_box = best_box_overall
        detected_class_name = detected_class_name_overall
        image_id = image_id_overall

    obstacle_id_to_return = obstacle_id if obstacle_id else detected_class_name

    # Draw the bounding box and save the image
    if selected_box:
        x1, y1, x2, y2 = selected_box.xyxy[0].cpu().numpy()

        # Draw the bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

        # Create labels
        label_line_1 = f"{detected_class_name}"
        label_line_2 = f"Image id={image_id}"

        # Set up font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # Calculate text size
        (w1, h1), _ = cv2.getTextSize(label_line_1, font, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(label_line_2, font, font_scale, thickness)
        text_width = max(w1, w2)
        text_height = h1 + h2 + 10

        # Position the text background
        text_offset_x = int(x1)
        text_offset_y = int(y1) - text_height if int(y1) - text_height > 0 else int(y1) + text_height

        # Draw background rectangle for text
        cv2.rectangle(img, (text_offset_x, text_offset_y),
                      (text_offset_x + text_width, text_offset_y + text_height),
                      (0, 255, 0), cv2.FILLED)

        # Put the text
        cv2.putText(img, label_line_1, (text_offset_x, text_offset_y + h1),
                    font, font_scale, (0, 0, 0), thickness)
        cv2.putText(img, label_line_2, (text_offset_x, text_offset_y + h1 + h2 + 5),
                    font, font_scale, (0, 0, 0), thickness)

        # Save the annotated image to the 'predictions' folder
        filename = os.path.join('predictions', f"annotated_image_{int(time.time())}.jpg")
        cv2.imwrite(filename, img)
        print(f"Image saved to {filename}")

    # Return the JSON response
    return {
        "obstacle_id": obstacle_id_to_return,
        "image_id": image_id
    }


def stitch_image():
    """
    Stitches images from the 'predictions' folder into a grid that adjusts to the number of images.
    Saves the stitched image with no unnecessary empty space.
    """
    imgFolder = 'predictions'  # Folder containing images
    stitchedPath = os.path.join(imgFolder, f'stitched-{int(time.time())}.jpeg')

    # Get all the images in the predictions folder
    imgPaths = glob.glob(os.path.join(imgFolder, "*.jpg"))
    print(f"Found images for stitching: {imgPaths}")  # Debugging line

    if not imgPaths:
        print("No images found to stitch.")
        return None

    # Open the images and ensure they have the same size
    images = [Image.open(x) for x in imgPaths]
    image_width, image_height = images[0].size

    # Dynamically determine the layout based on the number of images
    num_images = len(images)
    images_per_row = min(4, num_images)  # Max 4 images per row, but fewer if fewer images
    num_rows = (num_images + images_per_row - 1) // images_per_row  # Calculate rows needed

    # Calculate the total width and height of the stitched image
    total_width = images_per_row * image_width
    total_height = num_rows * image_height

    # Create a new blank image with the calculated dimensions
    stitched_img = Image.new('RGB', (total_width, total_height))

    # Paste each image into the stitched image in the correct position
    x_offset = 0
    y_offset = 0

    for i, img in enumerate(images):
        stitched_img.paste(img, (x_offset, y_offset))

        # Move to the next column
        x_offset += image_width

        # If we've reached the end of the row, move to the next row
        if (i + 1) % images_per_row == 0:
            x_offset = 0  # Reset x_offset to start a new row
            y_offset += image_height  # Move y_offset down to the next row

    # Save the stitched image
    stitched_img.save(stitchedPath)
    print(f"Stitched image saved at: {stitchedPath}")  # Debugging line

    return stitched_img

# Function to stitch images from the 'own_results' folder
def stitch_image_own():
    """
    Stitches the images in the 'own_results' folder together and saves the stitched image.
    """
    imgFolder = 'own_results'
    stitchedPath = os.path.join(imgFolder, f'stitched-{int(time.time())}.jpeg')

    imgPaths = glob.glob(os.path.join(imgFolder, "annotated_image_*.jpg"))
    if not imgPaths:
        print("No images found to stitch.")
        return None

    imgTimestamps = [imgPath.split("_")[-1][:-4] for imgPath in imgPaths]
    sortedByTimeStampImages = sorted(zip(imgPaths, imgTimestamps), key=lambda x: x[1])

    images = [Image.open(x[0]) for x in sortedByTimeStampImages]
    width, height = zip(*(i.size for i in images))
    total_width = sum(width)
    max_height = max(height)
    stitchedImg = Image.new('RGB', (total_width, max_height))
    x_offset = 0

    for im in images:
        stitchedImg.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    if not os.path.exists(imgFolder):
        os.makedirs(imgFolder)
    stitchedImg.save(stitchedPath)

    print(f"Stitched image saved at: {stitchedPath}")  # Debugging line
    return stitchedImg

# Flask route for image prediction
@app.route('/image', methods=['POST'])
def image_predict():
    try:
        # Check if the file is in the request
        if 'file' not in request.files:
            return jsonify({"error": "Missing file in request"}), 400

        file = request.files['file']
        filename = file.filename

        # Save the image file to a specific location
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Get obstacle_id from request arguments if provided
        obstacle_id = request.form.get('obstacle_id', None)

        # Call the predict_image_week_9 function with the file path, model, and obstacle_id
        result = predict_image_week_9(file_path, model, obstacle_id=obstacle_id)

        # Optional: Remove the uploaded file if not needed
        os.remove(file_path)

        return jsonify(result)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

# Route to trigger stitching from the 'predictions' folder
@app.route('/stitch', methods=['GET'])
def stitch():
    img = stitch_image()
    img2 = stitch_image_own()
    return jsonify({"result": "Stitching completed"})

# Flask route to check the status
@app.route('/status', methods=['GET'])
def status():
    return jsonify({"result": "ok"})

# Ensure necessary directories exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')
    print("Created 'uploads' directory.")

if not os.path.exists('predictions'):
    os.makedirs('predictions')
    print("Created 'predictions' directory.")

if __name__ == '__main__':
    print(f"Current working directory: {os.getcwd()}")
    app.run(host='0.0.0.0', port=5001, debug=True)
