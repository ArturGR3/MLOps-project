## End to End ML Deployment Using AutoGluon and OpenFE

The goal of this project to create a template of fast MVP (Minimal Viable Product) deployment with open source AutoML framework [Autogluon](https://auto.gluon.ai/stable/index.html) and auto feature engineering framework [OpenFE](https://github.com/IIIS-Li-Group/OpenFE) followed with some of the best practices learned during MLOps Zoomcamp. Due to automated nature, it should take minimal effort to adjust this project to a given data-set.

This project consists in 2 parts: 

1. [Part 1](part_1_building_the_model.md) covers steps of getting data and building the model.

2. [Part 2](part_2_deployment_visualization.md) covers model deployment as web service and monitoring.

### Key Features:
--- 

- Cloud-based Development: Built on Google Cloud VM using a free trial account
- Experiment Tracking: Utilizes MLflow Server on AWS EC2 with artifact storage in S3 bucket
- Modular Design: Custom [modules](modules/) for core functionality
- Orchestration:
    - [Part 1](part_1_building_the_model.md): [Jupyter notebook](MLOps_part_1_model_building.ipynb) for explanatory purposes
    - [Part 2](part_2_deployment_visualization.md): Fully dockerized and orchestrated using [Makefile](web_service_mlflow_visualiztion/Makefile)
- Monitoring: Basic monitoring implemented with Grafana
- Testing: Includes [unit tests](tests/unit_tests/) and [integration tests](tests/integration_test/) 