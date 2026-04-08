# Commands to rebuild and deploy the Fast API app to Azure Container Apps. Run these in WSL terminal from the root of the project.
az acr login --name salarypredictorscm

docker build -f dockerfile_api -t salary-predictor .

docker tag salary-predictor salarypredictorscm.azurecr.io/salary-predictor:v1

docker push salarypredictorscm.azurecr.io/salary-predictor:v1

az containerapp update \
  --name salary-predictor-app \
  --resource-group salary-predictor-rg \
  --image salarypredictorscm.azurecr.io/salary-predictor:v1