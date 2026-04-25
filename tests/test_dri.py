from core.diagnostic_reliability import reliability_gate


def test_reliability_gate_returns_status():
    result = reliability_gate(
        {"quality_score": 0.9},
        {"agreement_score": 0.9, "fused_confidence": 0.95, "lesion_trust_multiplier": 1.0},
        {"overlap_score": 0.8},
        {"score": 0.2},
    )
    assert result["acceptance_status"] in {"Accepted", "Caution", "Specialist Review Required"}
    assert "dri" in result
