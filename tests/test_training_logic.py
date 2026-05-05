import os
import sys
import unittest
from unittest import mock
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_single_gpu as t


def make_label(pathology_idx: int | None) -> torch.Tensor:
    """Create one PATH_COLS label vector with a single pathology turned on."""
    lab = torch.zeros(len(t.PATH_COLS), dtype=torch.float32)
    if pathology_idx is None:
        lab[0] = 1.0
    else:
        lab[pathology_idx] = 1.0
    return lab


class DummyTokens(dict):
    def to(self, device):
        return DummyTokens({k: v.to(device) for k, v in self.items()})


class DummyTokenizer:
    def __call__(self, captions, **kwargs):
        ids = []
        for cap in captions:
            ids.append(int(str(cap).split(":")[1]))
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(1)
        attention_mask = torch.ones_like(input_ids)
        return DummyTokens({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })


class DummyModel:
    def __init__(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor):
        self.image_embeds = image_embeds
        self.text_embeds = text_embeds
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def __call__(self, images, input_ids, attention_mask):
        img_ids = images[:, 0, 0, 0].round().long()
        txt_ids = input_ids[:, 0].long()
        return (
            self.image_embeds[img_ids],
            self.text_embeds[txt_ids],
            torch.tensor(1.0, dtype=torch.float32, device=images.device),
        )


class TrainingLogicTests(unittest.TestCase):
    def test_patient_split_has_no_leakage_and_is_deterministic(self):
        df = pd.DataFrame({
            "patient_id": [f"p{i // 2}" for i in range(20)],
            "image_id": [f"img_{i}.png" for i in range(20)],
        })
        train_a = t.patient_split(df, "train", seed=42)
        val_a = t.patient_split(df, "val", seed=42)
        test_a = t.patient_split(df, "test", seed=42)
        train_b = t.patient_split(df, "train", seed=42)

        pats_train = set(train_a["patient_id"])
        pats_val = set(val_a["patient_id"])
        pats_test = set(test_a["patient_id"])

        self.assertEqual(train_a["patient_id"].tolist(), train_b["patient_id"].tolist())
        self.assertTrue(pats_train.isdisjoint(pats_val))
        self.assertTrue(pats_train.isdisjoint(pats_test))
        self.assertTrue(pats_val.isdisjoint(pats_test))

    def test_labels_to_disease_vecs_merges_cardiomediastinum_into_cardiomegaly(self):
        labels = torch.zeros(2, len(t.PATH_COLS), dtype=torch.float32)
        labels[0, 1] = 1.0
        labels[1, 2] = 1.0
        disease_vecs = t.labels_to_disease_vecs(labels)

        self.assertEqual(disease_vecs.shape, (2, len(t.CHEXPERT_COLS)))
        self.assertTrue(torch.equal(disease_vecs[:, 0], torch.tensor([1.0, 1.0])))

    def test_build_positive_mask_uses_patient_id(self):
        mask = t.build_positive_mask(["p1", "p2", "p1"], torch.device("cpu"))
        expected = torch.tensor([
            [True, False, True],
            [False, True, False],
            [True, False, True],
        ])
        self.assertTrue(torch.equal(mask.cpu(), expected))

    def test_recall_at_k_hits_expected_topk(self):
        sim = torch.tensor([
            [0.9, 0.8, 0.1],
            [0.1, 0.3, 0.95],
        ], dtype=torch.float32)
        gt = torch.tensor([
            [True, False, False],
            [False, True, False],
        ])
        metrics = t.recall_at_k(sim, gt, ks=(1, 2, 3))
        self.assertAlmostEqual(metrics["R@1"], 50.0)
        self.assertAlmostEqual(metrics["R@2"], 100.0)
        self.assertAlmostEqual(metrics["R@3"], 100.0)

    def test_iuxray_dataset_loads_fields_and_labels(self):
        row = {
            "patient_id": "p1",
            "image_id": "sample.png",
            "org_caption": "normal report",
            "projection": "Frontal",
        }
        for col in t.PATH_COLS:
            row[col] = 0.0
        row["No Finding"] = 1.0
        df = pd.DataFrame([row])
        ds = t.IUXrayDataset(df, "unused_dir", t.get_val_transform())

        fake_img = Image.new("RGB", (8, 8), color=(64, 64, 64))
        with mock.patch("train_single_gpu.Image.open", return_value=fake_img):
            item = ds[0]

        self.assertEqual(item["pid"], "p1")
        self.assertEqual(item["proj"], "f")
        self.assertEqual(item["caption"], "normal report")
        self.assertEqual(tuple(item["labels"].shape), (14,))
        self.assertEqual(tuple(item["image"].shape), (3, 8, 8))

    def test_evaluate_reports_cluster_hits_when_strict_misses(self):
        num_samples = 10
        embed_dim = 10
        text_embeds = torch.eye(embed_dim, dtype=torch.float32)
        image_embeds = torch.zeros_like(text_embeds)

        samples = []
        pathology_cols = [3, 4, 5, 6, 7]
        for pair_idx in range(5):
            a = 2 * pair_idx
            b = a + 1
            pathology = pathology_cols[pair_idx]
            image_embeds[a] = torch.nn.functional.normalize(0.8 * text_embeds[b] + 0.6 * text_embeds[a], dim=0)
            image_embeds[b] = torch.nn.functional.normalize(0.8 * text_embeds[a] + 0.6 * text_embeds[b], dim=0)

            samples.append({
                "image": torch.full((3, 2, 2), float(a)),
                "caption": f"id:{a}",
                "pid": f"p{a}",
                "proj": "f",
                "labels": make_label(pathology),
            })
            samples.append({
                "image": torch.full((3, 2, 2), float(b)),
                "caption": f"id:{b}",
                "pid": f"p{b}",
                "proj": "l",
                "labels": make_label(pathology),
            })

        loader = DataLoader(samples, batch_size=5, shuffle=False, collate_fn=t.collate_fn)
        model = DummyModel(image_embeds, text_embeds)
        tokenizer = DummyTokenizer()

        strict_i2t, strict_t2i, cluster_i2t, cluster_t2i, sr1, cr1 = t.evaluate(
            model=model,
            loader=loader,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )

        self.assertAlmostEqual(strict_i2t["R@1"], 0.0)
        self.assertAlmostEqual(strict_t2i["R@1"], 0.0)
        self.assertAlmostEqual(strict_i2t["R@5"], 100.0)
        self.assertAlmostEqual(cluster_i2t["R@1"], 100.0)
        self.assertAlmostEqual(cluster_t2i["R@1"], 100.0)
        self.assertAlmostEqual(sr1, 0.0)
        self.assertAlmostEqual(cr1, 100.0)


if __name__ == "__main__":
    unittest.main()
