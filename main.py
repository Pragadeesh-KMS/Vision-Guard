from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
!pip install 'git+https://github.com/huggingface/transformers.git' -q
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
from google.colab import files
import io
import matplotlib.pyplot as plt
# Upload an image file
uploaded = files.upload()

# Select the uploaded image
file_name = list(uploaded.keys())[0]
image = Image.open(io.BytesIO(uploaded[file_name]))

# Initialize the processor and model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm")

# Process the uploaded image
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Convert outputs (bounding boxes and class logits) to COCO API
# Let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

# Display detection results
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )

# Display the image with bounding boxes around detected objects
plt.figure(figsize=(8, 8))
plt.imshow(image)
ax = plt.gca()

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    x_min, y_min, x_max, y_max = box
    w, h = x_max - x_min, y_max - y_min
    rect = plt.Rectangle((x_min, y_min), w, h, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    ax.text(x_min, y_min, f"{model.config.id2label[label.item()]} {round(score.item(), 3)}",
            bbox=dict(facecolor='white', alpha=0.7))

plt.axis('off')
plt.show()
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Create a copy of the original image to modify
blurred_image = image.copy()

# List of terms to check for violation. You  can change as per your wish and tune the prob score also.
violation_terms = [
    "Explicit", "nude", "brutality", "pornography", "violence", "hate", "racism", "terrorism",
    "abuse", "harassment", "offensive", "obscene", "drug", "illegal", "harmful", "threat",
    "bullying", "intolerance", "sexually explicit", "graphic content", 
    "something else", "something else", "something else"  # Repeat for additional coverage and do not remove the attributes "something else"
] # 3 "something else" is must to split the score among them if there are any other objects found

# Loop through the detected objects and perform CLIP analysis for each
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    x_min, y_min, x_max, y_max = map(int, box)  # Convert box coordinates to integers

    # Crop the detected object using coordinates
    cropped_image = image.crop((x_min, y_min, x_max, y_max))

    # Perform CLIP analysis on the cropped image for violating attributes
    inputs = processor(text=violation_terms, images=cropped_image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = torch.nn.functional.softmax(logits_per_image, dim=1)
    probs_list = probs.detach().numpy().tolist()[0]

    # Find the maximum score among all attribute scores
    max_score = max(probs_list)

    # Check if the maximum score for the object exceeds the threshold
    if max_score > 0.88:
        # Create a black rectangle to cover the violating region
        black_rectangle = Image.new("RGB", (x_max - x_min, y_max - y_min), color="black")
        blurred_image.paste(black_rectangle, (x_min, y_min))  # Place the black rectangle over the violating object

        print(f"Object at coordinates {box} has a maximum score of {max_score} which exceeds the given value")

# Display the final image with modifications
display(blurred_image)
