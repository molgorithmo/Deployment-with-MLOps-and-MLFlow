# Deployment-with-MLOps-and-MLFlow

This repository serves as a comprehensive guide and demonstration of deploying a machine learning project using MLFlow, Dagshub, AWS S3, and MLOps principles.

# Comparative Analysis: 
Compare multiple machine learning models (e.g., Linear Regression, Support Vector Regressor) using MLFlow UI to visualize scores and parameters.

The following snipshots shows how comparitive analysis can be conducted using MLFLow dashboard:

- Parameters and metrics of the models
    ![](https://github.com/molgorithmo/Deployment-with-MLOps-and-MLFlow/blob/main/imgs/MLFLow_params_metrics.png)

- Tabular view of the results
    ![](https://github.com/molgorithmo/Deployment-with-MLOps-and-MLFlow/blob/main/imgs/Tabular%20view.png)

- Model overview on dashboard
    ![](https://github.com/molgorithmo/Deployment-with-MLOps-and-MLFlow/blob/main/imgs/MLFlow_models.png)

- Graphs on model metrics
    ![](https://github.com/molgorithmo/Deployment-with-MLOps-and-MLFlow/blob/main/imgs/MLFlow_SVR.png)

# Features
- **MLFlow Integration:** Utilize MLFlow to track experiments, log parameters, metrics, and artifacts, and manage model versions.
- **Dagshub Collaboration:** Leverage Dagshub for version control, collaboration, and tracking changes across machine learning projects.
- **AWS S3 Integration:** Store and retrieve artifacts, datasets, and model artifacts using AWS S3, ensuring scalability, durability, and accessibility.
- **MLOps Best Practices:** Follow MLOps best practices to streamline the machine learning lifecycle, including model development, deployment, and monitoring.
- **Demonstration Project:** Machine learning project is provided to demonstrate the deployment pipeline using the aforementioned technologies.

## Getting Started
To get started with deploying your machine learning project using MLOps and MLFlow, follow these steps:

- Clone this repository to your local machine:

```console
git clone https://github.com/molgorithmo/Deployment-with-MLOps-and-MLFlow.git
```

- Install the necessary dependencies. You may use virtual environments or containerization tools like Docker to manage dependencies cleanly.

- Set up your MLFlow tracking server and configure the environment variables accordingly.

- Connect your project to Dagshub to enable version control and collaboration features.

- Configure AWS S3 credentials and bucket to store artifacts and datasets.

- Customize the sample machine learning project or integrate the provided deployment pipeline into your existing project.

- Run the deployment pipeline and monitor the progress using MLFlow UI and Dagshub.
