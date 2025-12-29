"""Unit tests for streamlit_tabs module.

Tests the data preparation functions (not the Streamlit rendering).
"""

from haoline.streamlit_tabs import (
    format_bytes,
    format_number,
    generate_cli_command,
    prepare_details_data,
    prepare_layer_table,
    prepare_model_info_table,
    prepare_op_distribution,
    prepare_quantization_data,
)


class MockMetadata:
    """Mock InspectionReport.metadata."""

    def __init__(
        self,
        ir_version: int = 8,
        producer_name: str | None = "pytorch",
        opsets: dict | None = None,
    ):
        self.ir_version = ir_version
        self.producer_name = producer_name
        # Use None check, not truthiness (empty dict should stay empty)
        self.opsets = opsets if opsets is not None else {"": 17}


class MockParamCounts:
    """Mock param counts."""

    def __init__(self, total: int = 1000000):
        self.total = total


class MockFlopCounts:
    """Mock flop counts."""

    def __init__(self, total: int = 5000000):
        self.total = total


class MockMemoryEstimates:
    """Mock memory estimates."""

    def __init__(
        self,
        peak_activation_bytes: int = 1024 * 1024,
        model_size_bytes: int = 2 * 1024 * 1024,
    ):
        self.peak_activation_bytes = peak_activation_bytes
        self.model_size_bytes = model_size_bytes


class MockGraphSummary:
    """Mock graph summary."""

    def __init__(
        self,
        num_nodes: int = 100,
        op_type_counts: dict | None = None,
    ):
        self.num_nodes = num_nodes
        # Use None check, not truthiness (empty dict should stay empty)
        self.op_type_counts = (
            op_type_counts if op_type_counts is not None else {"Conv": 20, "ReLU": 15, "Add": 10}
        )


class MockBlock:
    """Mock detected block."""

    def __init__(
        self,
        name: str = "encoder_block_0",
        block_type: str = "TransformerBlock",
        nodes: list | None = None,
        params: int | None = None,
    ):
        self.name = name
        self.block_type = block_type
        self.nodes = nodes or ["node1", "node2", "node3"]
        self.params = params


class MockRiskSignal:
    """Mock risk signal."""

    def __init__(
        self,
        id: str = "RISK001",
        severity: str = "medium",
        description: str = "Test risk",
    ):
        self.id = id
        self.severity = severity
        self.description = description


class MockReport:
    """Mock InspectionReport."""

    def __init__(
        self,
        metadata: MockMetadata | None = None,
        param_counts: MockParamCounts | None = None,
        flop_counts: MockFlopCounts | None = None,
        memory_estimates: MockMemoryEstimates | None = None,
        graph_summary: MockGraphSummary | None = None,
        detected_blocks: list | None = None,
        risk_signals: list | None = None,
    ):
        self.metadata = metadata or MockMetadata()
        self.param_counts = param_counts or MockParamCounts()
        self.flop_counts = flop_counts or MockFlopCounts()
        self.memory_estimates = memory_estimates or MockMemoryEstimates()
        self.graph_summary = graph_summary or MockGraphSummary()
        self.detected_blocks = detected_blocks
        self.risk_signals = risk_signals


class MockLayer:
    """Mock layer for layer summary."""

    def __init__(
        self,
        name: str = "conv1",
        op_type: str = "Conv",
        params: int = 1000,
        flops: int = 50000,
        memory_bytes: int = 4096,
        pct_params: float = 10.0,
        pct_flops: float = 5.0,
        output_shapes: list | None = None,
    ):
        self.name = name
        self.op_type = op_type
        self.params = params
        self.flops = flops
        self.memory_bytes = memory_bytes
        self.pct_params = pct_params
        self.pct_flops = pct_flops
        self.output_shapes = output_shapes or ["1x64x32x32"]


class MockLayerSummary:
    """Mock layer summary."""

    def __init__(self, layers: list | None = None):
        self.layers = layers or [MockLayer()]


class MockLintResult:
    """Mock quantization lint result."""

    def __init__(
        self,
        readiness_score: float = 85.0,
        warnings: list | None = None,
        unsupported_ops: set | None = None,
        accuracy_sensitive_ops: set | None = None,
        layer_risk_scores: list | None = None,
    ):
        self.readiness_score = readiness_score
        self.warnings = warnings
        self.unsupported_ops = unsupported_ops
        self.accuracy_sensitive_ops = accuracy_sensitive_ops
        self.layer_risk_scores = layer_risk_scores


class MockLayerRisk:
    """Mock layer risk score."""

    def __init__(
        self,
        layer_name: str = "softmax_0",
        op_type: str = "Softmax",
        risk_score: float = 0.8,
        reason: str = "Softmax is accuracy-sensitive",
    ):
        self.layer_name = layer_name
        self.op_type = op_type
        self.risk_score = risk_score
        self.reason = reason


class MockAdvice:
    """Mock quantization advice."""

    def __init__(
        self,
        strategy: str | None = None,
        qat_workflow: list | None = None,
    ):
        self.strategy = strategy or "Use INT8 for Conv layers"
        self.qat_workflow = qat_workflow or []


# =============================================================================
# Format function tests
# =============================================================================


class TestFormatNumber:
    """Tests for format_number()."""

    def test_small_number(self):
        assert format_number(500) == "500"

    def test_thousands(self):
        assert format_number(1500) == "1.50K"

    def test_millions(self):
        assert format_number(1_500_000) == "1.50M"

    def test_billions(self):
        assert format_number(1_500_000_000) == "1.50B"

    def test_trillions(self):
        assert format_number(1_500_000_000_000) == "1.50T"

    def test_zero(self):
        assert format_number(0) == "0"


class TestFormatBytes:
    """Tests for format_bytes()."""

    def test_bytes(self):
        assert format_bytes(500) == "500 B"

    def test_kilobytes(self):
        assert format_bytes(1500) == "1.50 KB"

    def test_megabytes(self):
        assert format_bytes(1_500_000) == "1.50 MB"

    def test_gigabytes(self):
        assert format_bytes(1_500_000_000) == "1.50 GB"

    def test_terabytes(self):
        assert format_bytes(1_500_000_000_000) == "1.50 TB"


# =============================================================================
# Data preparation tests
# =============================================================================


class TestPrepareModelInfoTable:
    """Tests for prepare_model_info_table()."""

    def test_basic_info(self):
        report = MockReport()
        props, metrics = prepare_model_info_table("test_model.onnx", report)

        assert len(props) == 4
        assert props[0]["Property"] == "Model"
        assert "`test_model.onnx`" in props[0]["Value"]

        assert len(metrics) == 4
        assert "1,000,000" in metrics[0]["Value"]  # params

    def test_missing_producer(self):
        report = MockReport(metadata=MockMetadata(producer_name=None))
        props, _ = prepare_model_info_table("model.onnx", report)

        producer_prop = next(p for p in props if p["Property"] == "Producer")
        assert producer_prop["Value"] == "Unknown"

    def test_empty_opsets(self):
        report = MockReport()
        report.metadata = MockMetadata(opsets={})
        props, _ = prepare_model_info_table("model.onnx", report)

        opset_prop = next(p for p in props if p["Property"] == "Opset")
        assert opset_prop["Value"] == "Unknown"

    def test_none_counts(self):
        report = MockReport()
        report.param_counts = None
        report.flop_counts = None
        report.memory_estimates = None
        _, metrics = prepare_model_info_table("model.onnx", report)

        assert metrics[0]["Value"] == "0"  # params
        assert metrics[1]["Value"] == "0"  # flops


class TestPrepareOpDistribution:
    """Tests for prepare_op_distribution()."""

    def test_basic_distribution(self):
        report = MockReport()
        data = prepare_op_distribution(report)

        assert len(data) == 3  # Conv, ReLU, Add
        assert data[0]["Operator"] == "Conv"  # Highest count first
        assert data[0]["Count"] == 20

    def test_top_n_limit(self):
        op_counts = {f"Op{i}": 100 - i for i in range(20)}
        report = MockReport(graph_summary=MockGraphSummary(op_type_counts=op_counts))

        data = prepare_op_distribution(report, top_n=5)
        assert len(data) == 5

    def test_empty_op_counts(self):
        report = MockReport()
        report.graph_summary = MockGraphSummary(op_type_counts={})
        data = prepare_op_distribution(report)
        assert data == []

    def test_none_graph_summary(self):
        report = MockReport()
        report.graph_summary = None
        data = prepare_op_distribution(report)
        assert data == []


class TestPrepareLayerTable:
    """Tests for prepare_layer_table()."""

    def test_basic_layer(self):
        layer_summary = MockLayerSummary()
        data = prepare_layer_table(layer_summary)

        assert len(data) == 1
        assert data[0]["Name"] == "conv1"
        assert data[0]["Op Type"] == "Conv"
        assert "1.00K" in data[0]["Parameters"]

    def test_redact_names(self):
        layer_summary = MockLayerSummary()
        data = prepare_layer_table(layer_summary, redact_names=True)

        assert data[0]["Name"] == "layer_0"

    def test_long_name_truncation(self):
        layer = MockLayer(name="a" * 50)
        layer_summary = MockLayerSummary(layers=[layer])
        data = prepare_layer_table(layer_summary)

        assert len(data[0]["Name"]) == 33  # 30 chars + "..."

    def test_max_layers_limit(self):
        layers = [MockLayer(name=f"layer_{i}") for i in range(200)]
        layer_summary = MockLayerSummary(layers=layers)
        data = prepare_layer_table(layer_summary, max_layers=50)

        assert len(data) == 50

    def test_zero_values(self):
        layer = MockLayer(params=0, flops=0, memory_bytes=0, pct_params=0, pct_flops=0)
        layer_summary = MockLayerSummary(layers=[layer])
        data = prepare_layer_table(layer_summary)

        assert data[0]["Parameters"] == "-"
        assert data[0]["FLOPs"] == "-"
        assert data[0]["Memory"] == "-"

    def test_empty_output_shapes(self):
        layer = MockLayer()
        layer.output_shapes = []  # Explicitly set empty
        layer_summary = MockLayerSummary(layers=[layer])
        data = prepare_layer_table(layer_summary)

        assert data[0]["Output Shape"] == "-"

    def test_none_layer_summary(self):
        data = prepare_layer_table(None)
        assert data == []

    def test_empty_layers(self):
        layer_summary = MockLayerSummary()
        layer_summary.layers = []  # Explicitly set empty
        data = prepare_layer_table(layer_summary)
        assert data == []


class TestPrepareQuantizationData:
    """Tests for prepare_quantization_data()."""

    def test_basic_data(self):
        lint_result = MockLintResult()
        data = prepare_quantization_data(lint_result)

        assert data["score"] == 85.0
        assert data["warnings"] == []
        assert data["unsupported_ops"] == []

    def test_with_warnings(self):
        lint_result = MockLintResult(warnings=["Warning 1", "Warning 2"])
        data = prepare_quantization_data(lint_result)

        assert len(data["warnings"]) == 2
        assert "Warning 1" in data["warnings"]

    def test_with_unsupported_ops(self):
        lint_result = MockLintResult(unsupported_ops={"CustomOp", "MyOp"})
        data = prepare_quantization_data(lint_result)

        assert "CustomOp" in data["unsupported_ops"]
        assert "MyOp" in data["unsupported_ops"]

    def test_with_sensitive_ops(self):
        lint_result = MockLintResult(accuracy_sensitive_ops={"Softmax", "LayerNorm"})
        data = prepare_quantization_data(lint_result)

        assert "Softmax" in data["sensitive_ops"]

    def test_with_advice(self):
        lint_result = MockLintResult()
        advice = MockAdvice(strategy="Use INT8", qat_workflow=["Step 1", "Step 2"])
        data = prepare_quantization_data(lint_result, advice)

        # strategy + first 2 qat_workflow items = 3 recommendations
        assert len(data["recommendations"]) == 3
        assert data["recommendations"][0] == "Use INT8"

    def test_with_layer_risks(self):
        lint_result = MockLintResult(layer_risk_scores=[MockLayerRisk()])
        data = prepare_quantization_data(lint_result)

        assert len(data["layer_risks"]) == 1
        assert data["layer_risks"][0]["Layer"] == "softmax_0"


class TestPrepareDetailsData:
    """Tests for prepare_details_data()."""

    def test_basic_data(self):
        report = MockReport()
        data = prepare_details_data(report)

        assert "blocks" in data
        assert "op_types" in data
        assert "risk_signals" in data

    def test_with_blocks(self):
        blocks = [MockBlock(name="block1"), MockBlock(name="block2")]
        report = MockReport(detected_blocks=blocks)
        data = prepare_details_data(report)

        assert len(data["blocks"]) == 2
        assert data["blocks"][0]["name"] == "block1"

    def test_block_with_params(self):
        blocks = [MockBlock(params=50000)]
        report = MockReport(detected_blocks=blocks)
        data = prepare_details_data(report)

        assert data["blocks"][0]["params"] == 50000

    def test_with_risk_signals(self):
        risks = [MockRiskSignal(id="R1"), MockRiskSignal(id="R2")]
        report = MockReport(risk_signals=risks)
        data = prepare_details_data(report)

        assert len(data["risk_signals"]) == 2
        assert data["risk_signals"][0]["id"] == "R1"

    def test_op_types_sorted(self):
        report = MockReport()
        data = prepare_details_data(report)

        # Should be sorted by count descending
        counts = [op["Count"] for op in data["op_types"]]
        assert counts == sorted(counts, reverse=True)

    def test_blocks_limit(self):
        blocks = [MockBlock(name=f"block_{i}") for i in range(20)]
        report = MockReport(detected_blocks=blocks)
        data = prepare_details_data(report)

        assert len(data["blocks"]) == 15  # Max 15


class TestGenerateCliCommand:
    """Tests for generate_cli_command() - Task 41.7.4."""

    def test_basic_command(self):
        cmd = generate_cli_command("model.onnx")
        assert "python -m haoline" in cmd
        assert "model.onnx" in cmd

    def test_with_hardware(self):
        cmd = generate_cli_command("model.onnx", hardware="rtx4090")
        assert "--hardware rtx4090" in cmd

    def test_with_auto_hardware(self):
        cmd = generate_cli_command("model.onnx", hardware="auto")
        assert "--hardware auto" in cmd

    def test_with_batch_size(self):
        cmd = generate_cli_command("model.onnx", batch_size=8)
        assert "--batch-size 8" in cmd

    def test_default_batch_size_not_included(self):
        cmd = generate_cli_command("model.onnx", batch_size=1)
        assert "--batch-size" not in cmd

    def test_html_output(self):
        cmd = generate_cli_command("model.onnx", output_format="html")
        assert "--out-html" in cmd
        assert "model_report.html" in cmd

    def test_json_output(self):
        cmd = generate_cli_command("model.onnx", output_format="json")
        assert "--out-json" in cmd
        assert "model_report.json" in cmd

    def test_md_output(self):
        cmd = generate_cli_command("model.onnx", output_format="md")
        assert "--out-md" in cmd
        assert "model_report.md" in cmd

    def test_include_graph(self):
        cmd = generate_cli_command("model.onnx", include_graph=True, output_format="html")
        assert "--include-graph" in cmd

    def test_no_include_graph(self):
        cmd = generate_cli_command("model.onnx", include_graph=False, output_format="html")
        assert "--include-graph" not in cmd

    def test_full_command(self):
        cmd = generate_cli_command(
            "resnet50.onnx",
            hardware="a100",
            batch_size=16,
            include_graph=True,
            output_format="html",
        )
        assert "python -m haoline" in cmd
        assert "resnet50.onnx" in cmd
        assert "--hardware a100" in cmd
        assert "--batch-size 16" in cmd
        assert "--include-graph" in cmd
        assert "--out-html resnet50_report.html" in cmd
