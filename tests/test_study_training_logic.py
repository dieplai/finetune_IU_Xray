import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_single_gpu as base
import train_proposed as study


def make_row(patient_id: str, image_id: str, projection: str, caption: str, pathology_col: str | None):
    row = {
        "patient_id": patient_id,
        "image_id": image_id,
        "projection": projection,
        "org_caption": caption,
    }
    for col in base.PATH_COLS:
        row[col] = 0.0
    if pathology_col is None:
        row["No Finding"] = 1.0
    else:
        row[pathology_col] = 1.0
    return row


class StudyTrainingLogicTests(unittest.TestCase):
    def test_study_dataset_groups_rows_by_patient(self):
        df = pd.DataFrame([
            make_row("p1", "img_f.png", "Frontal", "report-1", "Edema"),
            make_row("p1", "img_l.png", "Lateral", "report-1", "Edema"),
            make_row("p2", "img2_f.png", "Frontal", "report-2", None),
        ])
        ds = study.StudyIUXrayDataset(
            df=df,
            img_dir="unused",
            transform=base.get_val_transform(),
            train_mode=False,
        )
        self.assertEqual(len(ds), 2)

        fake_img = Image.new("RGB", (8, 8), color=(128, 128, 128))
        with mock.patch("train_proposed.Image.open", return_value=fake_img):
            item0 = ds[0]
            item1 = ds[1]

        self.assertEqual(item0["pid"], "p1")
        self.assertEqual(item0["caption"], "report-1")
        self.assertEqual(tuple(item0["images"].shape), (2, 3, 8, 8))
        self.assertTrue(torch.equal(item0["view_mask"], torch.tensor([True, True])))
        self.assertEqual(item1["pid"], "p2")
        self.assertEqual(item1["caption"], "report-2")
        self.assertTrue(torch.equal(item1["view_mask"], torch.tensor([True, False])))

    def test_build_cluster_mask_marks_shared_pathology_and_both_normal(self):
        labels = torch.zeros(3, len(base.PATH_COLS), dtype=torch.float32)
        labels[0, base.PATH_COLS.index("Edema")] = 1.0
        labels[1, base.PATH_COLS.index("Edema")] = 1.0
        labels[2, 0] = 1.0
        dv = base.labels_to_disease_vecs(labels)

        mask = study.build_cluster_mask(dv)
        self.assertTrue(mask[0, 1].item())
        self.assertTrue(mask[1, 0].item())
        self.assertFalse(mask[0, 2].item())
        self.assertTrue(mask[2, 2].item())

    def test_study_batch_sampler_fills_last_batch(self):
        df = pd.DataFrame([
            make_row("p1", "img1.png", "Frontal", "r1", "Edema"),
            make_row("p2", "img2.png", "Frontal", "r2", "Edema"),
            make_row("p3", "img3.png", "Frontal", "r3", "Edema"),
        ])
        ds = study.StudyIUXrayDataset(
            df=df,
            img_dir="unused",
            transform=base.get_val_transform(),
            train_mode=False,
        )
        sampler = study.StudyBatchSampler(ds, batch_size=2, seed=42)
        batches = list(iter(sampler))
        self.assertEqual(len(batches), 2)
        self.assertTrue(all(len(batch) == 2 for batch in batches))

    def test_clinical_supervised_contrastive_uses_non_normal_overlap(self):
        dv = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # p1
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # p2 same cluster
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # p3 different cluster
        ], dtype=torch.float32)
        loss_fn = study.ClinicalSupervisedContrastiveLoss()

        logits = torch.tensor([
            [0.30, 0.34, 0.10],
            [0.33, 0.31, 0.10],
            [0.10, 0.09, 0.35],
        ], dtype=torch.float32)

        self.assertGreater(loss_fn(logits, dv).item(), 0.0)

        no_overlap = torch.eye(3, len(base.CHEXPERT_COLS), dtype=torch.float32)
        self.assertAlmostEqual(loss_fn(logits, no_overlap).item(), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
