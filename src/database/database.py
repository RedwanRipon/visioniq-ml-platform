import json
import os

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from src.database.models import Base, Dataset, Experiment, Model
from src.utils.config import PROJECT_ROOT
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_engine():
    # DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/visioniq for PostgreSQL;
    # otherwise a local SQLite file, so the project works without a database server.
    default = f"sqlite:///{(PROJECT_ROOT / 'data' / 'metadata' / 'visioniq.db').as_posix()}"
    (PROJECT_ROOT / "data" / "metadata").mkdir(parents=True, exist_ok=True)
    return create_engine(os.environ.get("DATABASE_URL", default))


def record_final_model():
    engine = get_engine()
    Base.metadata.create_all(engine)                       # creates the tables if missing
    quality = json.loads((PROJECT_ROOT / "reports" / "data_quality_report.json").read_text())
    meta = json.loads((PROJECT_ROOT / "models" / "best_model_metadata.json").read_text())
    test = json.loads((PROJECT_ROOT / "reports" / "test_metrics.json").read_text())

    with Session(engine) as s:
        dataset = Dataset(name="cifar10", version="cifar-10-python",
                          sample_count=quality["train"]["samples"] + quality["test"]["samples"],
                          quality_score=None, status=quality["train"]["status"])
        s.add(dataset)
        s.flush()                                          # gives dataset an id
        exp = Experiment(dataset_id=dataset.id, mlflow_run_id=meta["mlflow_run_id"],
                         model_name=meta["model_name"], accuracy=test["accuracy"],
                         f1_score=test["f1_macro"])
        s.add(exp)
        s.flush()
        s.add(Model(experiment_id=exp.id, model_name=meta["model_name"],
                    version=meta["model_version"], artifact_path="models/best_model.pth"))
        s.commit()

        for table in (Dataset, Experiment, Model):
            for row in s.scalars(select(table)):
                logger.info("%s: %s", table.__tablename__,
                            {c.name: getattr(row, c.name) for c in table.__table__.columns})


if __name__ == "__main__":
    record_final_model()