# 1. Push the image to Azure Container Registry
# 2. Create a Containerapps Job
# 3. Configure the CRON Schedule
# 4. Assign Managed Identity
#    Storage Blob Data Contributor on salaryprdata (so it can read and write pipeline data)
#    AzureML Data Scientist on salary-predictor-ml (so it can log runs and register models)
# 5. Set environment variables
    The job needs MLFLOW_TRACKING_URI and ADZUNA_APP_ID/ADZUNA_APP_KEY injected — same pattern as the API container.

# Build the pipeline dockerfile
docker build -f dockerfile_pipeline -t salary-predictor-pipeline .

# Login to ACR
az acr login --name salarypredictorscm

# Tag the image
docker tag salary-predictor-pipeline salarypredictorscm.azurecr.io/salary-predictor-pipeline:v1

# Push
docker push salarypredictorscm.azurecr.io/salary-predictor-pipeline:v1



# Create the azure containerapps job. Relies on environment we set up previously. Has a 10 hour timeout - the longest the pipeline has taken is 4.5 hours. 
az containerapp job create \
  --name salary-predictor-pipeline-job \
  --resource-group salary-predictor-rg \
  --environment salary-predictor-env \
  --trigger-type Schedule \
  --cron-expression "0 18 * * *" \
  --replica-timeout 36000 \
  --replica-retry-limit 1 \
  --parallelism 1 \
  --replica-completion-count 1 \
  --image salarypredictorscm.azurecr.io/salary-predictor-pipeline:v1 \
  --registry-server salarypredictorscm.azurecr.io \
  --cpu 1.0 \
  --memory 2.0Gi


# Give the container a managed identity that we can then use to authenticate it to other services it needs to connect to
az containerapp job identity assign \
  --name salary-predictor-pipeline-job \
  --resource-group salary-predictor-rg \
  --system-assigned

# Storage Blob Data Contributor on salaryprdata
az role assignment create \
  --role "Storage Blob Data Contributor" \
  --assignee 6e72b567-3b40-46aa-8ba6-752f533a2ead \
  --scope $(az storage account show --name salaryprdata --resource-group salary-predictor-rg --query id -o tsv)


# AzureML Data Scientist on the ML workspace
az role assignment create \
  --role "AzureML Data Scientist" \
  --assignee 6e72b567-3b40-46aa-8ba6-752f533a2ead \
  --scope $(az ml workspace show --name salary-predictor-ml --resource-group salary-predictor-rg --query id -o tsv)

  # Add environmental variables to the pipeline containerapps job
  az containerapp job update \
  --name salary-predictor-pipeline-job \
  --resource-group salary-predictor-rg \
  --set-env-vars \
    MLFLOW_TRACKING_URI="azureml://australiaeast.api.azureml.ms/mlflow/v1.0/subscriptions/d9e61a78-9a8f-4d79-b9bd-8509b394cf9a/resourceGroups/salary-predictor-rg/providers/Microsoft.MachineLearningServices/workspaces/salary-predictor-ml" \
    ADZUNA_APP_ID="your-app-id" \
    ADZUNA_APP_KEY="your-app-key" \
    GIT_PYTHON_REFRESH="quiet" \
    PYTHONUNBUFFERED="1"


################## Testing ##################

# Manually trigger a run of the containerapps job
az containerapp job start \
  --name salary-predictor-pipeline-job \
  --resource-group salary-predictor-rg

# Watch the logs 
az containerapp job logs show \
  --name salary-predictor-pipeline-job \
  --resource-group salary-predictor-rg --follow
