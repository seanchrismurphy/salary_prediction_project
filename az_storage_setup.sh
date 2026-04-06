# Create the storage account
az storage account create \
  --name salaryprdata \
  --resource-group salary-predictor-rg \
  --location australiaeast \
  --sku Standard_LRS \
  --kind StorageV2 \
  --allow-blob-public-access false


  # Create a container inside it for the pipeline data
az storage container create \
  --name pipeline-data \
  --account-name salaryprdata \
  --auth-mode login

# Assign the right credentials
  az role assignment create \
  --role "Storage Blob Data Contributor" \
  --assignee $(az ad signed-in-user show --query id -o tsv) \
  --scope $(az storage account show --name salaryprdata --resource-group salary-predictor-rg --query id -o tsv)