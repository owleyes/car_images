# Car Images Project

This project focuses on processing and analyzing car images.

## Setup Instructions

### Prerequisites
- Python 3.13
- Conda (recommended)
- Git

### Using Conda (Recommended)

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

2. Run the main script:
```bash
python generate_damaged_cars.py
```

### Troubleshooting

#### Common Issues
1. **CUDA not found**: Ensure you have CUDA installed if you want to use GPU acceleration.
2. **Package version conflicts**: Try installing packages one by one if there are version conflicts.
3. **Memory issues**: Reduce batch sizes or use CPU if you encounter memory errors.

#### Package Notes
- `torch`: For deep learning operations
- `transformers`: For NLP and computer vision models
- `scikit-learn`: For traditional machine learning tasks
- `Pillow`: For image processing
- `numpy`: For numerical operations
- `pandas`: For data manipulation

### Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
