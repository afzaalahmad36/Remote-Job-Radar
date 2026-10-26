from pydantic import BaseModel


class HealthOut(BaseModel):
    status: str
    env: str
