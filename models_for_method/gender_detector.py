import numpy as np
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from config import GeneratorConfig

class GenderDetector:
    '''
    Класс для определения пола человека на изображении
    '''
    def __init__(self):
        self.model_name = GeneratorConfig.gender_detector_model_name
        self.model = SiglipForImageClassification.from_pretrained(self.model_name)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.id2label = {
            0: "female",
            1: "male"
        }

    def predict_gender(self, image):
        try:
            inputs = self.processor(images=image, return_tensors="pt")

            # Предсказание
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                predicted_class = torch.argmax(logits, dim=1).item()

            return self.id2label[predicted_class]

        except FileNotFoundError:
            return f"Ошибка: Файл '{image_path}' не найден"
        except Exception as e:
            return f"Ошибка при обработке изображения: {e}"
