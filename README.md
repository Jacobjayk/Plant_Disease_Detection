# Plant Disease Detection

This project utilizes deep learning to detect plant diseases from images of plant leaves. It includes Jupyter notebooks for training and testing the model, as well as a Streamlit web interface where users can upload images to get predictions.

## Files
- `main.py`: Streamlit app for plant disease detection.
- `test_plant_disease.ipynb`: Notebook for testing the model.
- `train_plant_disease.ipynb`: Notebook for training the model.
- `trained_plant_disease_model.keras`: The trained Keras model.
- `training_hist.json`: Training history.
- `home_page.jpeg`: Image for the home page.
- `README.md`: This file.
- `requirements.txt`: List of dependencies.

## Setup Instructions

1. **Clone the Repository**
    - Clone this repository to your local machine:
    ```sh
    git clone https://github.com/Jacobjayk/Plant_Disease_Detection.git
    cd Plant_Disease_Detection
    ```

2. **Install Dependencies**
    - Create a virtual environment and install the required dependencies:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Prepare the Dataset**
    - Download the [New Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from kaggle and extract it and copy the folders 'test', 'train', and 'valid' into the project directory.

4. **Data Preparation and Model Training**
    - Run the `train_plant_disease.ipynb` file in Jupyter Notebook to train the model and save the trained model:
    ```sh
    jupyter notebook train_plant_disease.ipynb
    ```

5. **Testing the Model**
    - After training, you can test the model using the `test_plant_disease.ipynb` notebook to evaluate its performance:
    ```sh
    jupyter notebook test_plant_disease.ipynb
    ```

6. **Run the Streamlit App**
    - Start the Streamlit app to use the web interface for plant disease detection:
    ```sh
    streamlit run main.py
    ```

## Usage

1. **Train the Model:** Follow the instructions in the `train_plant_disease.ipynb` notebook to train the model with your dataset.

2. **Test the Model:** Use the `test_plant_disease.ipynb` notebook to test the trained model and evaluate its performance.

3. **Run the Web App:** Start the Streamlit app using the command above and upload images of plant leaves to get disease predictions.

## Notes

- Ensure you have all dependencies installed as listed in `requirements.txt`.
- The [New Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) must be organized correctly for the data preparation and training notebooks to function properly.
- The trained model is saved as `trained_plant_disease_model.keras` and can be loaded by the Streamlit app for making predictions.

## Acknowledgement

This project was inspired by the need to provide accessible tools for plant disease detection, leveraging deep learning technologies to aid farmers and researchers in managing plant health more effectively. Special thanks to the creators of the [New Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) dataset and the open-source community for their contributions.

For any issues or contributions, please refer to the [GitHub repository](https://github.com/Jacobjayk/Plant_Disease_Detection).