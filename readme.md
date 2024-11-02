# Image Preprocessing with JSON Configuration in TensorFlow.js

This repository provides a simple setup for preprocessing images in JavaScript using TensorFlow.js, with configuration settings stored in `preprocessing.json`. This setup is ideal for ensuring consistent preprocessing across different environments, such as transferring a Python-based preprocessor into JavaScript.

## Files

- **preprocessing.json**: Configuration file that specifies preprocessing steps such as color mode, resizing, and normalization.
- **preprocess.js**: JavaScript file to load and apply the preprocessing settings in TensorFlow.js.

## `preprocessing.json` Configuration

The `preprocessing.json` file allows you to set the preprocessing parameters. Here’s a breakdown of each field:

```json
{
  "preprocess_image": {
    "color_mode": "grayscale",       // Converts the image to grayscale if set to "grayscale"
    "resize_shape": [48, 48],        // Resizes the image to the given width and height
    "normalize": true,               // Normalizes pixel values if true
    "scaling_factor": 255.0          // Divides pixel values by this factor during normalization
  },
  "class_labels": [
    "Marah 😡", "Senang 😊", "Netral 😐", "Sedih 😢"
  ]
}
```

## Instructions

### 1. Setup

1. **Google drive clone link model**:
   
   link download [Download model](https://drive.google.com/uc?id=1YQT-g3gAZqWS1dRwumvZKiFsREsmxr-J)
   

2. **Install TensorFlow.js** if you haven’t already:
   ```bash
   npm install @tensorflow/tfjs
   ```

### 2. Load and Apply Preprocessing in JavaScript

Use the provided JavaScript code (in `preprocess.js`) to load the configuration and preprocess the image.

#### Code Example

### 3. Explanation of the Code

1. **Grayscale Conversion**: Converts the image to grayscale if the `color_mode` is set to `"grayscale"`.
2. **Resize**: Resizes the image to `[48, 48]` or any dimensions set in `resize_shape`.
3. **Normalization**: Divides the pixel values by `scaling_factor` (255.0) to keep values between 0 and 1, which is a common preprocessing step.
4. **Batch Dimension**: Expands dimensions to make the shape `[1, 48, 48, 1]`, adding a batch dimension for compatibility with TensorFlow.js models.

### Class Labels

The `preprocessing.json` file also includes a `class_labels` field, which provides the labels for each emotion class:
- **Marah 😡**
- **Senang 😊**
- **Netral 😐**
- **Sedih 😢**

In JavaScript, load `class_labels` by accessing `config.class_labels` after loading `preprocessing.json`. The `getClassLabels` function in the example will retrieve and display these labels.

### Sample Code to Display Class Labels


## Usage Notes

- **TensorFlow.js**: This setup requires TensorFlow.js for image processing, so ensure it's included in your project.
- **Image Element**: Ensure that the `imageElement` variable references an HTML `<img>` element that you want to preprocess.
- **JSON Configuration**: Modify `preprocessing.json` if you need to change preprocessing settings like color mode or image size.

## Additional Resources

- [TensorFlow.js Documentation](https://js.tensorflow.org/api/latest/)
- [Image Preprocessing in TensorFlow.js](https://www.tensorflow.org/js/guide)