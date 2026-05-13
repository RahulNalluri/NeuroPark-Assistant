🅿️ Smart Parking AI System
A deep learning project that detects vehicle types from images and automatically assigns them to the right parking zone — built to solve the real problem of unorganized parking in malls and commercial spaces.

💡 Why I Built This
Anyone who's been to a mall knows the chaos — SUVs squeezed into compact spots, motorcycles blocking car bays, and attendants manually directing traffic. We wanted to fix that with AI.
This system takes a photo of a vehicle, figures out what type it is, and tells it exactly where to park. Simple idea, powerful execution.

🚗 What It Can Detect
VehicleParking ZoneSUVZone A — Wide baysPickup TruckZone B — Extra long baysSedanZone C — Standard baysHatchbackZone D — Compact baysMotorcycleZone F — Dedicated bays

🧠 How It Works

Upload a vehicle image
The AI model (EfficientNetV2B3) classifies it
If confidence is above 85% → parking zone is assigned
If confidence is below 85% → attendant is alerted


📊 Model Performance
MetricScoreOverall Accuracy99.33% Precision0.99 Recall0.99 F1-Score0.99 Error Rate0.67%
Three out of five classes hit a perfect F1-score of 1.00.

🛠️ Tech Stack

Model — EfficientNetV2B3 (Transfer Learning)
Framework — TensorFlow / Keras
Web App — Streamlit
Training — Two-phase fine tuning (300×300 input)
Dataset — 6,600+ real-world Indian vehicle images


🚀 Getting Started
bash# Clone the repo
git clone https://github.com/yourusername/image_classification_1.git
cd image_classification_1

# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

📁 Project Structure
image_classification_1/
├── dataset/
│   ├── train/
│   └── validation/
├── model/
│   ├── car_model.h5
│   └── class_indices.json
├── image_classification_model.ipynb
├── app.py
├── requirements.txt
└── README.md

📦 Requirements
tensorflow
streamlit
numpy
Pillow
pandas
scikit-learn
matplotlib