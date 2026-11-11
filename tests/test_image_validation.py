"""Tests for the conservative ImageNet label screening rule."""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from predict import UnsupportedImageError, predict_image
from utils.image_validation import is_clearly_unrelated_image


class ImageValidationTests(unittest.TestCase):
    def test_recognized_unrelated_categories_are_rejected(self):
        labels = [
            "golden_retriever",
            "tabby",
            "soccer_player",
            "sports_car",
            "seashore",
        ]

        for label in labels:
            with self.subTest(label=label):
                self.assertTrue(is_clearly_unrelated_image(label, 0.8))

    def test_waste_related_labels_are_not_rejected(self):
        labels = ["water_bottle", "carton", "can_opener", "paper_towel", "ashcan"]

        for label in labels:
            with self.subTest(label=label):
                self.assertFalse(is_clearly_unrelated_image(label, 0.8))

    def test_weak_imagenet_score_is_not_enough_to_reject(self):
        self.assertFalse(is_clearly_unrelated_image("golden_retriever", 0.34))

    def test_rejected_image_never_reaches_waste_classifier(self):
        image_validator = Mock()
        image_validator.predict.return_value = np.zeros((1, 1000))
        waste_model = Mock()
        class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

        with patch(
            "predict.decode_predictions",
            return_value=[[("n02099601", "golden_retriever", 0.9)]],
        ):
            with self.assertRaises(UnsupportedImageError):
                self._predict_temporary_image(
                    waste_model, class_names, image_validator
                )

        waste_model.predict.assert_not_called()

    def test_supported_image_still_uses_six_class_model(self):
        image_validator = Mock()
        image_validator.predict.return_value = np.zeros((1, 1000))
        waste_model = Mock()
        waste_model.predict.return_value = np.array([[0, 0, 0, 0, 0.9, 0.1]])
        class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

        with patch(
            "predict.decode_predictions",
            return_value=[[("n04522168", "water_bottle", 0.8)]],
        ):
            probabilities, label, confidence = self._predict_temporary_image(
                waste_model, class_names, image_validator
            )

        self.assertEqual(label, "plastic")
        self.assertEqual(confidence, 0.9)
        self.assertEqual(len(probabilities), 6)
        waste_model.predict.assert_called_once()

    @staticmethod
    def _predict_temporary_image(waste_model, class_names, image_validator):
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = os.path.join(temporary_directory, "image.png")
            Image.new("RGB", (8, 8)).save(image_path)
            return predict_image(
                waste_model, image_path, class_names, image_validator
            )


if __name__ == "__main__":
    unittest.main()
