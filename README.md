
# Stock Prediction App

## 🚀 Overview
This Stock Prediction App utilizes advanced machine learning models to predict the required stock amounts for specific products on future dates. It leverages two robust models: **SARIMAX** and **LSTM**, to provide predictions and their respective accuracies.

## 📦 Features
- Predict stock requirements using historical data.
- Utilize SARIMAX and LSTM models for predictions.
- Interface for entering product names and selecting prediction dates.

## 🛠️ Setup
To run this project locally, follow these steps:

### Prerequisites
- Ensure you have Python installed on your machine. [Download Python](https://www.python.org/downloads/)
- It's recommended to use a virtual environment for Python projects.

### Installation
1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
1. Run the model setup script to initialize and train the models using the provided dataset. This will only need to be done once before the first run or after updating the dataset:
   ```bash
   python model.py
   ```
2. Start the Flask server to host the application:
   ```bash
   python app.py
   ```
3. Open the `index.html` file in a web browser to access the user interface.

## 🖥️ Using the Application
- Open the web interface provided by `index.html`.
- Enter a **Product Name** and a **Future Date** for which you want the stock prediction.
- Click **Predict** to receive stock predictions from both the SARIMAX and LSTM models along with their Root Mean Square Error (RMSE) accuracies.

## 🧪 Example Products for Testing
- Air Filter
- Alternator
- Brake Pads
- Engine oil
(Note: You can use any other product name from the dataset for realistic testing.)

## 📈 Model Information
- **SARIMAX**: Ideal for time series forecasting with seasonal patterns.
- **LSTM**: Great for sequences and predictions where long-term dependencies are crucial.

## 📝 Notes
- The accuracy of predictions can vary based on the quality and quantity of the data provided.
- Adjustments to model parameters may be necessary depending on the specific characteristics of the new dataset.
