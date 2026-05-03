from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: int = Field(..., ge=0, le=1)
    Dependents: int = Field(..., ge=0, le=1)
    tenure: int = Field(..., ge=0, description="Meses como cliente")
    PhoneService: int = Field(..., ge=0, le=1)
    MultipleLines: int = Field(..., ge=0, le=1)
    OnlineSecurity: int = Field(..., ge=0, le=1)
    OnlineBackup: int = Field(..., ge=0, le=1)
    DeviceProtection: int = Field(..., ge=0, le=1)
    TechSupport: int = Field(..., ge=0, le=1)
    StreamingTV: int = Field(..., ge=0, le=1)
    StreamingMovies: int = Field(..., ge=0, le=1)
    PaperlessBilling: int = Field(..., ge=0, le=1)
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: float = Field(..., ge=0)
    InternetService: str = Field(..., description="DSL | Fiber optic | No")
    Contract: str = Field(..., description="Month-to-month | One year | Two year")
    PaymentMethod: str = Field(
        ...,
        description="Electronic check | Mailed check | Bank transfer (automatic) | Credit card (automatic)",
    )


class PredictionOutput(BaseModel):
    churn_probability: float = Field(..., description="Probabilidade de churn (0 a 1)")
    churn_prediction: bool = Field(..., description="True se churn provável (prob >= 0.5)")
