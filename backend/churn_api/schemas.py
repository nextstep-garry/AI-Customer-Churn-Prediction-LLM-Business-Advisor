"""Request schemas. Ranges/categories come from ml.spec, the same source the batch cleaner uses."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .ml import spec

R = spec.RANGES


class CustomerIn(BaseModel):
    """One raw customer record. Unknown fields, NaN/Infinity, nulls and out-of-range values are rejected."""
    model_config = ConfigDict(extra="forbid")

    CreditScore: int = Field(ge=R["CreditScore"][0], le=R["CreditScore"][1])
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Female", "Male"]
    Age: int = Field(ge=R["Age"][0], le=R["Age"][1])
    Tenure: int = Field(ge=R["Tenure"][0], le=R["Tenure"][1])
    Balance: float = Field(ge=R["Balance"][0], le=R["Balance"][1], allow_inf_nan=False)
    NumOfProducts: int = Field(ge=R["NumOfProducts"][0], le=R["NumOfProducts"][1])
    HasCrCard: int = Field(ge=0, le=1)
    IsActiveMember: int = Field(ge=0, le=1)
    EstimatedSalary: float = Field(gt=R["EstimatedSalary"][0], le=R["EstimatedSalary"][1], allow_inf_nan=False)

    @field_validator("Geography", "Gender", mode="before")
    @classmethod
    def _normalise_case(cls, v):
        return v.strip().title() if isinstance(v, str) else v


class BatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[CustomerIn] = Field(min_length=1)
