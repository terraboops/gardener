"""Composable-DAG pipeline IR + loaders + executor."""
from .ir import Edge, Node, Pipeline
from .yaml_loader import load_pipeline_yaml, compose_pipeline
from .prose_parser import parse_pipeline_prose

__all__ = ["Edge", "Node", "Pipeline",
           "load_pipeline_yaml", "parse_pipeline_prose", "compose_pipeline"]
