from pydantic import BaseModel, ConfigDict, Field

# A single Telco record with features already label-encoded the same way
# ml_churn.data.encode() does (categoricals -> ints). Useful as a ready-to-send
# Swagger example. Edit the numbers in the UI to test other customers.
_EXAMPLE_INSTANCE = {
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 12,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": 1,
    "OnlineSecurity": 0,
    "OnlineBackup": 1,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 1,
    "StreamingMovies": 1,
    "Contract": 0,
    "PaperlessBilling": 1,
    "PaymentMethod": 2,
    "MonthlyCharges": 79.85,
    "TotalCharges": 958.2,
}


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "instances": [
                        _EXAMPLE_INSTANCE,
                        {**_EXAMPLE_INSTANCE, "tenure": 60, "Contract": 2,
                         "MonthlyCharges": 25.1, "TotalCharges": 1506.0},
                    ]
                }
            ]
        }
    )

    instances: list[dict[str, float]] = Field(
        ...,
        description="Batch of records; each maps feature name -> numeric value "
        "(features are expected pre-encoded, matching the model signature).",
    )


class PredictByIdRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"customer_ids": ["7590-VHVEG", "5575-GNVDE"]}]
        }
    )

    customer_ids: list[str] = Field(
        ...,
        description="Batch of entity keys (customerID). Features are fetched from "
        "the feature store; the caller does not send raw feature values.",
    )


class Prediction(BaseModel):
    churn: int = Field(..., description="Predicted label (1 = will churn).")
    churn_probability: float = Field(..., description="Probability of churn.")


class PredictResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model_name": "churn-lgbm",
                    "model_version": "3",
                    "predictions": [
                        {"churn": 1, "churn_probability": 0.7421},
                        {"churn": 0, "churn_probability": 0.1083},
                    ],
                }
            ]
        }
    )

    model_name: str
    model_version: str
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelMetadataResponse(BaseModel):
    name: str
    version: str
    run_id: str | None
    stage: str | None
    aliases: list[str]
    creation_timestamp: int
    features: list[str]
    metrics: dict[str, float]
    params: dict[str, str]
    loaded_at: str
