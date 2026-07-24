from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streamvad.data import GATE_CLASS_IDS, StreamVADStage1Dataset, StreamVADStage2GateDataset
from tools import build_streamvad_weak_supervision as builder
from tools.train_streamvad_stage1_lora import compute_event_token_indices


def _vadr1_row(*, video: str, anomaly_type: str, start: int | None = None, end: int | None = None) -> dict:
    row = {
        "source": "mock",
        "video": video,
        "anomaly_type": anomaly_type,
        "path": f"/mock/videos/{video}.mp4",
        "total_frames": 300,
        "think": "<step1>A quiet lobby with people walking normally.</step1><step2>A visible event occurs.</step2>",
        "answer": "<which>{}</which><what>A person is {}.</what><why>Visible action supports the label.</why>".format(
            "Normal" if anomaly_type == "Normal" else "Abnormal",
            "walking normally" if anomaly_type == "Normal" else "fighting",
        ),
    }
    if start is not None:
        row["start"] = start
    if end is not None:
        row["end"] = end
    return row


class StreamVADDataTests(unittest.TestCase):
    def test_stage1_target_uses_think_answer_format(self) -> None:
        args = _args()
        row = _vadr1_row(video="abn", anomaly_type="Fighting", start=90, end=150)

        stage1 = builder.make_stage1(row, args)

        self.assertIsNotNone(stage1)
        assert stage1 is not None
        self.assertTrue(stage1["target_text"].startswith("<think>\n"))
        self.assertIn("</think>\n<answer>\nAbnormal\n</answer>", stage1["target_text"])
        self.assertEqual(stage1["event_token_fractions"], [0.1, 0.5, 0.9])

    def test_stage1_skips_normal_rows(self) -> None:
        args = _args()
        row = _vadr1_row(video="normal", anomaly_type="Normal")

        self.assertIsNone(builder.make_stage1(row, args))

    def test_stage1_loader_rejects_event_outside_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_stage1.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "video": "/missing.mp4",
                        "clip_start": 0,
                        "clip_end": 1,
                        "event_start_sec": 1.5,
                        "event_end_sec": 2.0,
                        "target_text": "<think>\nBad event.\n</think>\n<answer>\nAbnormal\n</answer>",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "event range"):
                StreamVADStage1Dataset(path, require_video_exists=False)

    def test_event_token_indices_map_start_middle_end(self) -> None:
        indices = compute_event_token_indices(
            clip_start=1.0,
            clip_end=7.0,
            event_start=3.0,
            event_end=5.0,
            num_frames=13,
            fractions=[0.1, 0.5, 0.9],
        )

        self.assertEqual(indices, [4, 6, 8])

    def test_stage2_gate_action_uses_hold_trigger(self) -> None:
        args = _args()
        row = _vadr1_row(video="abn", anomaly_type="Fighting", start=90, end=150)

        chunks = builder.make_stage2(row, args)
        actions = {chunk["gate_action"] for chunk in chunks}

        self.assertIn("hold", actions)
        self.assertIn("trigger", actions)
        self.assertNotIn("silence", actions)
        self.assertNotIn("response", actions)
        for chunk in chunks:
            self.assertEqual(GATE_CLASS_IDS[chunk["gate_action"]], chunk["gate_label"])

    def test_stage2_loader_rejects_label_action_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_stage2.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "video": "/missing.mp4",
                        "chunk_start": 0,
                        "chunk_end": 1,
                        "gate_label": 1,
                        "gate_action": "hold",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "disagrees"):
                StreamVADStage2GateDataset(path, require_video_exists=False)

    def test_mock_builder_cli_writes_event_cot_and_hold_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "mock_vadr1.jsonl"
            output_dir = tmp_path / "out"
            rows = [
                _vadr1_row(video="normal", anomaly_type="Normal"),
                _vadr1_row(video="abnormal", anomaly_type="Fighting", start=90, end=150),
            ]
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "build_streamvad_weak_supervision.py"),
                    "--input-jsonl",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--train-ratio",
                    "0.5",
                    "--fps",
                    "30",
                ],
                check=True,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            stage1_rows = _read_jsonl(output_dir / "streamvad_stage1_train.jsonl")
            stage1_rows.extend(_read_jsonl(output_dir / "streamvad_stage1_val.jsonl"))
            stage2_rows = _read_jsonl(output_dir / "streamvad_stage2_train.jsonl")
            stage2_rows.extend(_read_jsonl(output_dir / "streamvad_stage2_val.jsonl"))

            self.assertEqual(len(stage1_rows), 1)
            self.assertEqual(stage1_rows[0]["answer"], "abnormal")
            self.assertTrue(stage1_rows[0]["target_text"].startswith("<think>\n"))
            self.assertLessEqual({row["gate_action"] for row in stage2_rows}, {"hold", "trigger", "ignore"})


def _args():
    return type(
        "Args",
        (),
        {
            "path_prefix_from": None,
            "path_prefix_to": None,
            "fps": 30.0,
            "pre_context_sec": 2.0,
            "post_context_sec": 2.0,
            "chunk_duration_sec": 1.0,
            "boundary_radius_sec": 1.0,
        },
    )()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    unittest.main()
