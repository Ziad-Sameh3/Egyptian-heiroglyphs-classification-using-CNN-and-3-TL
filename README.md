# Egyptian Hieroglyphs Classification Using CNN and Transfer Learning

This project demonstrates the use of a Convolutional Neural Network (CNN) and transfer learning to classify images of Egyptian hieroglyphs into 95 distinct classes. The implementation is done in Python using TensorFlow and Keras and is designed to work with a dataset containing annotated hieroglyph images.

## Features

- Cropping and preprocessing of images based on bounding box annotations.
- Transfer learning using pre-trained models for improved performance.
- Classification of hieroglyphs into 95 unique classes.
- Data augmentation to enhance model generalization.

## Dataset

The dataset consists of Egyptian hieroglyphs organized into three folders:

- **train/**: Training images.
- **valid/**: Validation images.
- **test/**: Test images.

Each folder contains a corresponding `_annotations.csv` file with the following columns:

- `filename`: Name of the image file.
- `class`: Class label of the hieroglyph.
- `width`, `height`: Dimensions of the image.
- `xmin`, `ymin`, `xmax`, `ymax`: Bounding box coordinates.

## Prerequisites

- Python 3.7+
- Jupyter Notebook
- Kaggle environment (optional for running the notebook)

### Python Libraries

The following libraries are required to run the code:

- `numpy`
- `pandas`
- `tensorflow`
- `matplotlib`
- `opencv-python`

Install the dependencies using pip:

```bash
pip install numpy pandas tensorflow matplotlib opencv-python
```

## How to Use

1. **Download the Dataset**: Ensure the dataset is structured as described and contains the necessary annotations.
2. **Run the Notebook**:
   - Open the notebook `egyptian-heiroglyphs-classify-cnn-tl.ipynb` in Jupyter Notebook or Google Colab.
   - Ensure the dataset is uploaded and accessible in the environment.
3. **Preprocess Images**: The notebook includes code to crop and resize images to 224x224 pixels based on bounding box annotations.
4. **Train the Model**: Execute the training steps in the notebook. The model uses transfer learning with a pre-trained backbone such as ResNet or VGG.
5. **Evaluate the Model**: Use the test set to evaluate classification accuracy and other metrics.

## Results

The trained CNN achieves accurate classification across 95 hieroglyph classes. The notebook provides:

- Training and validation accuracy plots.
- Sample predictions with images.


## Future Improvements

- Incorporate additional data augmentation techniques.
- Experiment with different transfer learning backbones.
- Optimize hyperparameters for better performance.

## Author

**Ziad Sameh**\
Feel free to reach out for collaboration or questions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Acknowledgments

Special thanks to the Kaggle platform for providing an accessible environment for experimentation and learning.

