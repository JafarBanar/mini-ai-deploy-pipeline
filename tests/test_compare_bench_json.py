import json

from compare_bench_json import compare


def test_compare_bench_json(tmp_path):
    ref_path = tmp_path / "ref.json"
    cand_path = tmp_path / "cand.json"
    out_path = tmp_path / "out.json"

    ref = {
        "backend": "onnxruntime",
        "latency_ms": {"p50": 10.0, "p95": 20.0, "mean": 12.0},
    }
    cand = {
        "backend": "tensorrt",
        "latency_ms": {"p50": 5.0, "p95": 10.0, "mean": 6.0},
    }

    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    cand_path.write_text(json.dumps(cand), encoding="utf-8")

    result = compare(
        reference_path=str(ref_path),
        candidate_path=str(cand_path),
        reference_label="ort",
        candidate_label="trt",
        out_json=str(out_path),
    )

    assert result["summary"]["p95"]["speedup_x"] == 2.0
    assert out_path.exists()

