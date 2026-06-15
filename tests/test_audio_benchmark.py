from activation_benchmark.audio_benchmark import _aggregate, _paired_rows


def _row(condition, seed, accuracy, duration):
    return {
        "condition": condition,
        "seed": seed,
        "best_validation_accuracy": accuracy + 0.01,
        "test_accuracy": accuracy,
        "final_test_accuracy": accuracy - 0.01,
        "total_duration_seconds": duration,
    }


def test_audio_aggregate_preserves_condition_order():
    rows = [
        _row("gelu", 42, 0.80, 10),
        _row("periodic_gelu", 42, 0.82, 12),
        _row("direct_peuaf", 42, 0.78, 13),
        _row("evolved_peuaf", 42, 0.81, 20),
    ]

    aggregate = _aggregate(rows)

    assert [row["condition"] for row in aggregate] == [
        "gelu",
        "periodic_gelu",
        "direct_peuaf",
        "evolved_peuaf",
    ]
    assert aggregate[1]["test_accuracy_mean"] == 0.82


def test_audio_paired_rows_report_percentage_point_differences():
    rows = [
        _row("gelu", 42, 0.80, 10),
        _row("periodic_gelu", 42, 0.82, 12),
        _row("direct_peuaf", 42, 0.78, 13),
        _row("evolved_peuaf", 42, 0.81, 20),
    ]

    paired = _paired_rows(rows)

    assert paired[0]["periodic_gelu_minus_gelu"] == 2.0
    assert paired[0]["evolved_minus_direct"] == 3.0
