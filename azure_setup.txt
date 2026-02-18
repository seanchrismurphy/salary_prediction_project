Initial Setup
bash# Install Azure CLI (WSL)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Verify logged in
az account show

# Register required providers
az provider register --namespace Microsoft.ContainerRegistry
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights

# Verify registration (wait until all show "Registered")
az provider show --namespace Microsoft.ContainerRegistry --query registrationState
az provider show --namespace Microsoft.App --query registrationState
az provider show --namespace Microsoft.OperationalInsights --query registrationState
Create Azure Resources
bash# Create resource group
az group create \
  --name salary-predictor-rg \
  --location australiaeast

# Create container registry (replace <yourname> with something unique)
az acr create \
  --resource-group salary-predictor-rg \
  --name salarypredictorscm \
  --sku Basic

# Enable admin access on registry
az acr update \
  --name salarypredictorscm \
  --admin-enabled true

# Get registry credentials (save these)
az acr credential show --name salarypredictorscm
Build and Push Docker Image
bash# Tag image for Azure registry
sudo docker tag salary-predictor:v2 salarypredictorscm.azurecr.io/salary-predictor:v2

# Login to registry (use credentials from above)
sudo docker login salarypredictorscm.azurecr.io \
  --username <username> \
  --password <password>

# Push image
sudo docker push salarypredictorscm.azurecr.io/salary-predictor:v2

# Verify image is in registry
az acr repository show-tags --name salarypredictorscm --repository salary-predictor
Deploy Container App
bash# Create container app environment
az containerapp env create \
  --name salary-predictor-env \
  --resource-group salary-predictor-rg \
  --location australiaeast

# Create container app (use ACR credentials from earlier)
az containerapp create \
  --name salary-predictor-app \
  --resource-group salary-predictor-rg \
  --environment salary-predictor-env \
  --image salarypredictorscm.azurecr.io/salary-predictor:v2 \
  --target-port 8000 \
  --ingress external \
  --registry-server salarypredictorscm.azurecr.io \
  --registry-username <username> \
  --registry-password <password> \
  --cpu 0.5 \
  --memory 1.0Gi \
  --min-replicas 0 \
  --max-replicas 1

# Get public URL
az containerapp show \
  --name salary-predictor-app \
  --resource-group salary-predictor-rg \
  --query properties.configuration.ingress.fqdn \
  --output tsv
Test Deployment
bash# Replace <url> with output from previous command
curl -X POST https://<url>/predict -H "Content-Type: application/json" -d '{"job_title": "Senior Data Scientist", "description": "Lead machine learning initiatives.", "contract_type": "permanent", "contract_time": "full_time", "category_label": "IT Jobs", "location_area_length": 3, "location_state": "New South Wales", "location_region": "Sydney", "location_city": "Sydney", "missing_long_lat": false, "longitude": 151.2093, "latitude": -33.8688}'
Useful Commands
bash# View logs from running container
az containerapp logs tail \
  --name salary-predictor-app \
  --resource-group salary-predictor-rg

# List all resources in resource group
az resource list \
  --resource-group salary-predictor-rg \
  --output table

# Delete everything when done
az group delete --name salary-predictor-rg --yes
```

**requirements-serve.txt** (for slim Docker image)
```
fastapi
uvicorn[standard]
pydantic
scikit-learn
numpy
joblib
