#  House Price Prediction using Multivariate Linear Regression

This project implements multivariate linear regression from scratch to predict house prices using features such as square footage and number of bedrooms.  
It follows the methodology taught in **Andrew Ng’s Machine Learning Specialization** and uses only **NumPy**, with no machine learning libraries.

---

##  Overview

The model predicts the price of a house given:
- Size in square feet
- Number of bedrooms

The process includes:
- Feature normalization
- Cost function computation (mean squared error)
- Gradient descent optimization
- Parameter learning and price prediction

---

##  Machine Learning Concepts Used

- Linear Regression (with multiple variables)  
- Feature Scaling (Normalization)  
- Vectorized Cost Function  
- Batch Gradient Descent  
- Hypothesis Function `h(x) = Xθ`  
- Manual Prediction (no ML libraries)

---

##  Files

- `house_price_predictor.py` – Main script with complete implementation  
- `README.md` – Project documentation (this file)

---

##  How to Run

1. Install dependencies:
   ```bash
   pip install numpy matplotlib
2. Run the script:
python house_price_predictor.py

---

## Future Ideas
- Add more features like location, number of bathrooms
- Use a larger real-world dataset
- Implement the normal equation method
- Add evaluation using a test set
- Visualize 3D regression surface

---

## Author
Alok Pudasaini
Civil Engineer

---

## Inspired By
Andrew Ng – Machine Learning Specialization (Coursera)

---