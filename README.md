# Car Images Project

This project focuses on detecting fake car damage images.

## Setup Instructions

### Prerequisites
- Python 3.13
- Conda (recommended)
- Git

### Setup Conda Environment

1. Create a new conda environment with Python 3.13:
```bash
conda create -n car_images python=3.13
conda activate car_images
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

### Project Structure
```
car_images/
├── requirements.txt      # Python dependencies
├── generate_damaged_cars.py  # Main script
├── .gitignore          # Git ignore patterns
└── README.md          # This file
```

### Running the Project

1. Ensure you're in the correct conda environment:
```bash
conda activate car_images
```

2. Run the image generation script:
```bash
python generate_damaged_cars.py
```

3. Run the image detector.ipynb notebook from your Jupyter environment
