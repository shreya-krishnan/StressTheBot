from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database.connection import Base


def utc_now():
    return datetime.now(timezone.utc)


class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True, autoincrement=True)
    rule_id = Column(String, nullable=False, index=True)
    experiment_status = Column(String, nullable=False, index=True)
    rule_status = Column(String, nullable=False)
    research_only = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    request_config = Column(Text, nullable=False)
    risk_summary = Column(Text, nullable=False)
    scenarios = relationship("Scenario", back_populates="experiment", cascade="all, delete-orphan")
    errors = relationship("ExecutionError", back_populates="experiment", cascade="all, delete-orphan")


class Scenario(Base):
    __tablename__ = "scenarios"
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    scenario_id = Column(String, nullable=False)
    rule_id = Column(String, nullable=False)
    patient_profile = Column(Text, nullable=False)
    rendered_prompt = Column(Text, nullable=True)
    experiment = relationship("Experiment", back_populates="scenarios")
    model_responses = relationship("ModelResponse", back_populates="scenario", cascade="all, delete-orphan")
    evaluations = relationship("Evaluation", back_populates="scenario", cascade="all, delete-orphan")
    risk_records = relationship("RiskRecord", back_populates="scenario", cascade="all, delete-orphan")


class ModelResponse(Base):
    __tablename__ = "model_responses"
    id = Column(Integer, primary_key=True, autoincrement=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=True)
    prompt = Column(Text, nullable=True)
    raw_response_text = Column(Text, nullable=True)
    response_status = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    extracted_data = Column(Text, nullable=True)
    scenario = relationship("Scenario", back_populates="model_responses")


class Evaluation(Base):
    __tablename__ = "evaluations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    rule_id = Column(String, nullable=False)
    status = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    matched_prohibited_patterns = Column(Text, nullable=False)
    missing_required_patterns = Column(Text, nullable=False)
    explanation = Column(Text, nullable=False)
    requires_human_review = Column(Boolean, nullable=False)
    scenario = relationship("Scenario", back_populates="evaluations")


class RiskRecord(Base):
    __tablename__ = "risk_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    rule_id = Column(String, nullable=False)
    status = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    patient_parameters = Column(Text, nullable=False)
    evidence = Column(Text, nullable=False)
    matched_prohibited_patterns = Column(Text, nullable=False)
    missing_required_patterns = Column(Text, nullable=False)
    explanation = Column(Text, nullable=False)
    requires_human_review = Column(Boolean, nullable=False)
    scenario = relationship("Scenario", back_populates="risk_records")


class ExecutionError(Base):
    __tablename__ = "execution_errors"
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    scenario_label = Column(String, nullable=True)
    error_type = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    experiment = relationship("Experiment", back_populates="errors")
