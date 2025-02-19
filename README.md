# 🎓 End-to-End Student Performance Prediction Project

## 📌 Project Overview
This project aims to predict student performance based on academic and socio-economic factors, helping educators and policymakers make data-driven decisions. The project covers:

- **Data Handling**: Ingestion and preprocessing of student data.
- **Model Training**: Experimentation with multiple regression models to predict math scores.
- **Prediction Pipeline**: Application of trained models to new data for predictions.
- **Web Application**: A Flask-based web interface for users to input student data and get predictions.
- **Deployment**: Utilization of Docker for containerizing the application and hosting on Azure Web Apps.

## 🛠 Tools and Technologies Used

### 🚀 Deployment & CI/CD
- **Amazon EC2 & GitHub Actions self-hosted runners**  
  - EC2 instances were configured as GitHub Actions self-hosted runners to provide a controlled CI/CD environment.
  - These runners executed build and test jobs, ensuring compatibility with production.
- **Amazon Elastic Container Registry (ECR)**  
  - Served as the repository for Docker images. Built images were pushed to ECR, ensuring the latest version was always available for deployment.
- **Azure Container Registry (ACR)**  
  - Used alongside AWS ECR for a multi-cloud image management strategy, allowing redundancy and Azure-based deployments.
- **Azure Web App**  
  - Hosted the final application by pulling the Docker image from ACR, ensuring high availability and easy scalability.

### 📊 Machine Learning
- **Regression Models**: 
  - Linear Regression, Decision Trees, Random Forests, XGBoost
- **Evaluation Metrics**:
  - R² Score, Mean Absolute Error (MAE)
