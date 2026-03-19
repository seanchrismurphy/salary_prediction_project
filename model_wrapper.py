import mlflow
import numpy as np

def predict_pipeline(bundle, df):
    """Transform df using fitted bundle and return predictions."""
    X_title = bundle["title_tfidf"].transform(df["title"].fillna(""))
    X_desc = bundle["desc_svd"].transform(
        bundle["desc_tfidf"].transform(df["full_description"].fillna(""))
    )
    X_structured = np.hstack(
        [
            bundle["ohe"].transform(df[bundle["cat_vars"]]),
            bundle["scaler"].transform(
                bundle["imputer"].transform(df[bundle["num_vars"]])
            ),
        ]
    )
    X = np.hstack([X_title.toarray(), X_desc, X_structured])
    return bundle["model"].predict(X)

# THis step actually registers the model so that it will show up on the 'Models' page of MLflow.
class SalaryPipelineWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib

        self.bundle = joblib.load(context.artifacts["pipeline_bundle"])

    def predict(self, context, model_input):
        return predict_pipeline(self.bundle, model_input)