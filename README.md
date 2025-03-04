# SafeScan - AI-Powered COVID-19 Detection

![SafeScan Logo](https://amirbekshomurodov.me/static/855a6e9a3a644df727b9e9024d5aeada/9298b/covid-detector.webp)

## 📌 Overview  
SafeScan is an AI-powered web application designed to detect COVID-19 using X-ray images. By leveraging deep learning techniques, it offers **fast, accurate, and accessible** diagnosis, reducing reliance on traditional PCR and antigen tests.

## 🔥 Features  
- ✅ AI-driven COVID-19 detection from X-ray images  
- ✅ Web-based interface for easy access  
- ✅ MobileNet deep learning model for high accuracy  
- ✅ Cloud deployment with **Heroku & AWS S3**  
- ✅ Supports real-time image analysis  

## 🛠️ Technologies Used  
- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Flask (Python)  
- **Model Architecture:** MobileNet  
- **Database:** AWS S3 for model storage  
- **Deployment:** Heroku, GitHub Actions (CI/CD)  

## 🚀 How It Works  
1. Users **upload an X-ray image** via the web interface  
2. The image is processed using a **deep learning model**  
3. The system predicts if the image **indicates COVID-19**  
4. Results are displayed instantly with **confidence scores**  

## ⚙️ Installation & Setup  
Follow these steps to set up the project locally:

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Amirbeek/COVID-19-Detector-Web-Application.git
cd SafeScan
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application  
```bash
python app.py
```

The web application will be available at `http://127.0.0.1:5000/`

## 📂 Project Structure  
```
SafeScan/
│── static/            # Static assets (CSS, JS, images)
│── templates/         # HTML templates
│── models/           # Trained AI models
│── app.py            # Flask application
│── requirements.txt  # Dependencies
└── README.md         # Project documentation
```

## 📜 License  
This project is licensed under the **MIT License**.

## 🤝 Acknowledgements  
Special thanks to **Brunel University London** and my supervisor **Webio Liu** for their guidance and support throughout this project.

---

🚀 **Contributions are welcome!** Feel free to fork this repository and submit a pull request. Let's make AI-powered diagnostics more accessible! 🌍
