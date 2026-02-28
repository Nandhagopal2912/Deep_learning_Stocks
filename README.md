# Deep_learning_Stocks 📈

A simple **LSTM-based stock price prediction model** using historical price data.  
This project reads stock data from a CSV, visualizes it, trains an LSTM neural network, and plots predictions vs actual values.

## 🧠 Overview

This repository contains a **deep learning model** that leverages a Long Short-Term Memory (LSTM) network to predict future stock closing prices based on past data. It uses TensorFlow/Keras for model building and training, and matplotlib/seaborn for visualization.

**Primary goal:** Build a basic deep learning model to forecast stock prices using historical data.

---

## 📁 Repository Structure
Deep_learning_Stocks/
├── .gitignore
├── requiremnets.txt
├── stock-prediction.py
└── stocks.csv

- **stock-prediction.py** – Python script with the LSTM model  
- **stocks.csv** – Sample dataset for training/testing  
- **requiremnets.txt** – Python dependencies

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- pandas & NumPy  
- scikit-learn  
- matplotlib & seaborn  
- LSTM (Recurrent Neural Networks)

---

## 🚀 Installation

1. **Clone the repository**

   bash
   git clone https://github.com/Nandhagopal2912/Deep_learning_Stocks.git
   cd Deep_learning_Stocks

2. ***Create a Vitrual Environment***
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows

3. ***Install Dependencied***
   pip install -r requiremnets.txt

4. ***Usage***
   python stock-prediction.py

   
