
# Vision Guard: Personalized Content Oversight Tool

## Colab Link

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dg0hPF_hn5CGHYcyWfAleBWXsrW4z_6-?usp=sharing)

## Overview

Vision Guard is an innovative content moderation tool utilizing state-of-the-art computer vision and language models to empower users with personalized content oversight. It enables users to filter objectionable or unwanted content from images based on their specified terms.

## Key Features

- **Object Identification:** Utilizes Facebook's DETR model for precise object detection and labeling within images.
- **Content Moderation:** Leverages OpenAI's CLIP model to assess identified objects against user-defined terms for moderation.
- **Personalization:** Allows users to input and modify their own terms, tailoring content filtration to individual preferences.

## Objective

The primary objective of Vision Guard is to provide a robust content moderation system that empowers users to control the content they view, ensuring a personalized and safe browsing experience.

## Workflow

1. **Object Detection:**
   - DETR model identifies and labels objects within uploaded images.
2. **Content Moderation:**
   - CLIP model evaluates identified objects against user-provided terms.
   - Objects matching specified terms are obscured or blacked out in the image.
3. **User Interface (Future Implementation):**
   - Development of an intuitive user interface (using computer vision techniques) for user interaction and input of moderation preferences.

## Getting Started

### Requirements

- Python 3.x
- Libraries: `transformers`, `Pillow`, `torch`, `opencv`, etc.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Pragadeesh-KMS/Vision-Guard.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Upload an image file to the system.
2. Run the provided scripts to perform object detection and content moderation.
3. View the moderated output image with objectionable content obscured.

### Future Development

- **System Integration:** Develop Vision Guard into a system software for broader usage.
- **Enhanced UI/UX:** Create an intuitive user interface for seamless user interaction.
- **Real-time Monitoring:** Implement real-time monitoring capabilities for ongoing content oversight.

## Roadmap

1. **Research & Development Phase:** Experimentation with DETR and CLIP models for object detection and content moderation.
2. **Implementation:** Code the proposed workflow and user interaction (GUI).
3. **Testing & Optimization:** Test the tool with various images and scenarios, optimize for performance.
4. **Future Enhancements:** Integrate additional features and scalability options.

## Contribution & Collaboration

Contributions, suggestions, and collaborations from the open-source community are welcome! Feel free to fork the repository and submit pull requests.

## Acknowledgments

- **Facebook's DETR:** 
- **OpenAI's CLIP:** 
