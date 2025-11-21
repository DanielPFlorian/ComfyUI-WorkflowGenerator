"""Tests for ComfyUI-WorkflowGenerator custom nodes."""

from nodes.node_validator_node import NodeValidatorNode
from nodes.pipeline_node import WGPipelineNode
from nodes.update_catalog_node import UpdateNodeCatalogNode
from nodes.workflow_builder_node import WorkflowBuilderNode
from nodes.workflow_generator_node import WorkflowGeneratorNode


def test_workflow_generator_schema():
    """Test that WorkflowGenerator node has correct schema."""
    schema = WorkflowGeneratorNode.define_schema()
    assert schema.node_id == "WorkflowGenerator"
    assert schema.category == "WorkflowGenerator"
    assert "instruction" in [inp.id for inp in schema.inputs]
    assert "workflow_edges" in [out.display_name for out in schema.outputs]


def test_node_validator_schema():
    """Test that NodeValidator node has correct schema."""
    schema = NodeValidatorNode.define_schema()
    assert schema.node_id == "NodeValidator"
    assert schema.category == "WorkflowGenerator"
    assert "workflow_edges" in [inp.id for inp in schema.inputs]
    assert "workflow_edges (refined)" in [out.display_name for out in schema.outputs]


def test_workflow_builder_schema():
    """Test that WorkflowBuilder node has correct schema."""
    schema = WorkflowBuilderNode.define_schema()
    assert schema.node_id == "WorkflowBuilder"
    assert schema.category == "WorkflowGenerator"
    assert "workflow_edges" in [inp.id for inp in schema.inputs]
    assert "workflow_json" in [out.display_name for out in schema.outputs]


def test_pipeline_schema():
    """Test that Pipeline node has correct schema."""
    schema = WGPipelineNode.define_schema()
    assert schema.node_id == "WG_Pipeline"
    assert schema.category == "WorkflowGenerator"
    assert "instruction" in [inp.id for inp in schema.inputs]
    assert len(schema.outputs) == 4  # workflow_edges, workflow_edges (refined), workflow_json, file_path


def test_update_catalog_schema():
    """Test that UpdateNodeCatalog node has correct schema."""
    schema = UpdateNodeCatalogNode.define_schema()
    assert schema.node_id == "WG_UpdateNodeCatalog"
    assert schema.category == "WorkflowGenerator"
    assert "catalog_directory" in [inp.id for inp in schema.inputs]
    assert "node_count" in [out.display_name for out in schema.outputs]
