#!/bin/bash
# Azure CLI Commands — Salary Predictor Setup
# Session 2: Azure ML Workspace + API Container Update
# Run these in order. Prerequisites: az login, az account show to verify subscription.

# ============================================================
# AZURE ML WORKSPACE
# ============================================================

# Install the Azure ML CLI extension (required for az ml commands)
az extension add -n ml

# Create the Azure ML workspace
# This automatically provisions: storage account, key vault, log analytics, app insights
az ml workspace create \
  --name salary-predictor-ml \
  --resource-group salary-predictor-rg \
  --location australiaeast

# Retrieve the MLflow tracking URI for the workspace
# Use this value as MLFLOW_TRACKING_URI in .env and Container App config
az ml workspace show \
  --name salary-predictor-ml \
  --resource-group salary-predictor-rg \
  --query mlflow_tracking_uri

# ============================================================
# MANAGED IDENTITY — API CONTAINER APP
# ============================================================

# Assign a system-managed identity to the Container App
# This allows the container to authenticate to Azure services without credentials
az containerapp identity assign \
  --name salary-predictor-app \
  --resource-group salary-predictor-rg \
  --system-assigned

# Get the principal ID of the managed identity (needed for role assignment below)
az containerapp identity show \
  --name salary-predictor-app \
  --resource-group salary-predictor-rg \
  --query principalId \
  --output tsv

# Grant the Container App's managed identity the AzureML Data Scientist role
# on the ML workspace — allows it to read models from the registry
# Replace <principal-id> with output from the command above
az role assignment create \
  --assignee <principal-id> \
  --role "AzureML Data Scientist" \
  --scope /subscriptions/d9e61a78-9a8f-4d79-b9bd-8509b394cf9a/resourceGroups/salary-predictor-rg/providers/Microsoft.MachineLearningServices/workspaces/salary-predictor-ml

# ============================================================
# REBUILD AND PUSH API CONTAINER IMAGE
# ============================================================

# Build the new image locally (--no-cache forces clean install of all packages)
sudo docker build --no-cache -t salary-predictor:v3 .

# Verify packages installed correctly in the image
sudo docker run --rm salary-predictor:v3 pip freeze

# Tag the image for the Azure Container Registry
sudo docker tag salary-predictor:v3 salarypredictorscm.azurecr.io/salary-predictor:v3

# Log in to ACR (get credentials from: az acr credential show --name salarypredictorscm)
sudo docker login salarypredictorscm.azurecr.io \
  --username <username> \
  --password <password>

# Push the image to ACR
sudo docker push salarypredictorscm.azurecr.io/salary-predictor:v3

# Verify the image is in the registry
az acr repository show-tags --name salarypredictorscm --repository salary-predictor

# ============================================================
# UPDATE CONTAINER APP
# ============================================================

# Update the Container App to use the new image
# Also sets the MLFLOW_TRACKING_URI environment variable
# Managed Identity and other env vars already set — persist across updates
az containerapp update \
  --name salary-predictor-app \
  --resource-group salary-predictor-rg \
  --image salarypredictorscm.azurecr.io/salary-predictor:v3 \
  --set-env-vars MLFLOW_TRACKING_URI="azureml://australiaeast.api.azureml.ms/mlflow/v1.0/subscriptions/d9e61a78-9a8f-4d79-b9bd-8509b394cf9a/resourceGroups/salary-predictor-rg/providers/Microsoft.MachineLearningServices/workspaces/salary-predictor-ml"

# ============================================================
# USEFUL DEBUGGING COMMANDS
# ============================================================

# List all resources in the resource group
az resource list \
  --resource-group salary-predictor-rg \
  --output table

# Stream live logs from the running Container App
az containerapp logs show \
  --name salary-predictor-app \
  --resource-group salary-predictor-rg \
  --tail 50 \
  --follow

# Test a prediction against the live API
curl -X POST https://<container-app-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"job_title": "Senior Data Scientist", "description": "Lead machine learning initiatives.", "contract_type": "permanent", "contract_time": "full_time", "category_label": "IT Jobs", "location_area_length": 3, "location_state": "New South Wales", "location_region": "Sydney", "location_city": "Sydney", "missing_long_lat": false, "longitude": 151.2093, "latitude": -33.8688}'

# Run container locally for testing (mounts local Azure credentials)
sudo docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI="azureml://australiaeast.api.azureml.ms/mlflow/v1.0/subscriptions/d9e61a78-9a8f-4d79-b9bd-8509b394cf9a/resourceGroups/salary-predictor-rg/providers/Microsoft.MachineLearningServices/workspaces/salary-predictor-ml" \
  -v /home/seancm/.azure:/root/.azure \
  salary-predictor:v3

# Run a single command inside a container image without starting the server
sudo docker run --rm salary-predictor:v3 <command>
# e.g. pip freeze, pip show azureml-mlflow, ls /root/.azure