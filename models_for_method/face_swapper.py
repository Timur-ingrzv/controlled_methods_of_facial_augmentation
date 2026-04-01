import os
import time
import base64
import re
import requests
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Optional, Tuple

from config import SwapperConfig

class FaceSwapper:
    """
    Класс для замены лица 
    """
    def __init__(
        self,
        selfie_path: str,
        mask_selfie_face_path: str,
        mask_selfie_photo_path: str,
        folder_a: str,
        folder_b: str,
        display_image: bool = False
    ):
        self.deployment_id = SwapperConfig.deployment_id
        self.token = SwapperConfig.token
        self.selfie_url = self._encode_image_to_base64_url(selfie_path)
        self.mask_selfie_face_url = self._encode_image_to_base64_url(mask_selfie_face_path)
        self.mask_selfie_photo_url = self._encode_image_to_base64_url(mask_selfie_photo_path)
        self.folder_a = folder_a
        self.folder_b = folder_b
        self.output_dir = SwapperConfig.output_dir
        self.display_image = display_image
        self.prompt = "white circle"
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _encode_image_to_base64_url(image_path: str) -> str:
        """Перевод изображения в нужный формат"""
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{image_data}"

    def _wait_for_completion(self, status_url: str, max_attempts: int = 100) -> str:
        """Ожидание завершения инференса"""
        attempt = 0
        status = "in_queue"
        while status not in ["completed", "succeeded", "failed"] and attempt < max_attempts:
            time.sleep(10)
            attempt += 1
            try:
                resp = requests.get(status_url, headers={"Authorization": f"Bearer {self.token}"})
                if resp.status_code != 200:
                    print(f"Ошибка получения статуса: {resp.status_code}")
                    return "failed"
                status_data = resp.json()
                status = status_data.get('status', 'unknown')
            except Exception as e:
                return "failed"
        return status

    def _download_result(self, result_url: str, save_path: str) -> bool:
        """Скачивание результата"""
        try:
            resp = requests.get(result_url, headers={"Authorization": f"Bearer {self.token}"})
            if resp.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(resp.content)

                if self.display_image:
                    from IPython.display import Image as IPImage, display
                    print(f"Превью {save_path}:")
                    display(IPImage(save_path))
                return True
            else:
                print(f"Ошибка скачивания {save_path}: {resp.status_code}")
                return False
        except Exception as e:
            print(f"Ошибка скачивания {save_path}: {e}")
            return False

    def _run_inference_step(
        self,
        target_image_url: str,
        source_image_url: str,
        source_mask_url: str,
        target_mask_url: str,
        output_path: str
    ) -> bool:
        """Выполнение шага для изображения"""
        payload = {
            "overrides": {
                "110": {"inputs": {"image": source_image_url}},
                "109": {"inputs": {"image": source_mask_url}},
                "108": {"inputs": {"image": target_image_url}},
                "111": {"inputs": {"image": target_mask_url}},
                "98":  {"inputs": {"prompt": self.prompt}},
                "99":  {"inputs": {"prompt": self.prompt}},
            }
        }

        try:
            response = requests.post(
                f"https://api.runcomfy.net/prod/v1/deployments/{self.deployment_id}/inference",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.token}"
                },
                json=payload
            )

            result = response.json()
            status_url = result.get('status_url')
            result_url = result.get('result_url')

            status = self._wait_for_completion(status_url)
            if status not in ["completed", "succeeded"]:
                print(f"Инференс завершился с ошибкой: {status}")
                return False

            result_resp = requests.get(result_url, headers={"Authorization": f"Bearer {self.token}"})
            if result_resp.status_code != 200:
                print(f"Ошибка получения результата: {result_resp.status_code}")
                return False

            result_data = result_resp.json()
            for node_id, node_data in result_data.get('outputs', {}).items():
                if node_id == '102' and node_data['images'][0]['type'] == 'output':
                    image_url = node_data['images'][0]['url']
                    return self._download_result(image_url, output_path)

        except Exception as e:
            print(e)

    def swap_faces_for_person(self, person_idx: int) -> Tuple[Optional[str], Optional[str]]:
        """
        Замена лиц для одного человека
        """
        src_image_step1 = os.path.join(self.folder_b, f"person_{person_idx+1}.jpeg")
        src_mask_step1 = os.path.join(self.folder_b, f"mask_person_{person_idx+1}.jpeg")
        src_image_step2 = os.path.join(self.folder_a, f"person_{person_idx+1}.jpeg")
        src_mask_step2 = os.path.join(self.folder_a, f"mask_person_{person_idx+1}.jpeg")

        src_url_step1 = self._encode_image_to_base64_url(src_image_step1)
        src_mask_url_step1 = self._encode_image_to_base64_url(src_mask_step1)
        src_url_step2 = self._encode_image_to_base64_url(src_image_step2)
        src_mask_url_step2 = self._encode_image_to_base64_url(src_mask_step2)

        # Шаг 1
        step1_output = os.path.join(self.output_dir, f"res_person_{person_idx}_step_1.jpeg")
        print(f"Шаг 1 для person_{person_idx}")
        success1 = self._run_inference_step(
            target_image_url=self.selfie_url,
            source_image_url=src_url_step1,
            source_mask_url=src_mask_url_step1,
            target_mask_url=self.mask_selfie_face_url,
            output_path=step1_output
        )
        if not success1:
            return None, None

        with open(step1_output, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
            step1_result_url = f"data:image/jpeg;base64,{image_data}"

        # Шаг 2
        step2_output = os.path.join(self.output_dir, f"res_person_{person_idx}_step_2.jpeg")
        print(f"Шаг 2 для person_{person_idx}")
        success2 = self._run_inference_step(
            target_image_url=step1_result_url,
            source_image_url=src_url_step2,
            source_mask_url=src_mask_url_step2,
            target_mask_url=self.mask_selfie_photo_url,
            output_path=step2_output
        )

        return step1_output, step2_output

    def swap_faces(self, n: int) -> List[Tuple[Optional[str], Optional[str]]]:
        """
        Перенос лица для изображений
        """
        results = []
        for i in range(1, n + 1):
            needed = [
                os.path.join(self.folder_a, f"person_{i}.jpeg"),
                os.path.join(self.folder_a, f"mask_person_{i}.jpeg"),
                os.path.join(self.folder_b, f"person_{i}.jpeg"),
                os.path.join(self.folder_b, f"mask_person_{i}.jpeg")
            ]
            if all(os.path.exists(p) for p in needed):
                step1, step2 = self.swap_faces_for_person(i-1)
                results.append((step1, step2))
            else:
                print(f"Пропуск person_{i}: отсутствуют некоторые файлы")
                results.append((None, None))
        return results

    @staticmethod
    def show_images_side_by_side(image_paths: List[str], titles: Optional[List[str]] = None):
        """Отображение изображений"""
        n_images = len(image_paths)
        fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
        if n_images == 1:
            axes = [axes]
        for i, path in enumerate(image_paths):
            img = Image.open(path)
            axes[i].imshow(img)
            axes[i].axis('off')
            if titles and i < len(titles):
                axes[i].set_title(titles[i])
        plt.tight_layout()
        plt.show()
