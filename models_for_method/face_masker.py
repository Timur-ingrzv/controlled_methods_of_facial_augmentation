import os
import sys 

sys.path.insert(0, '/content')

import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from config import MaskerConfig

class FaceMasker:
    """Класс для создания масок лиц с помощью SAM2"""

    def __init__(self, box_adj={'top': -70, 'left': -50, 'right': 0, 'bottom': 0},
                 borders=True):
        """
        Инициализация маскера
        """
        self.device = MaskerConfig.device
        self.sam2_checkpoint = MaskerConfig.sam2_checkpoint
        self.model_cfg = MaskerConfig.model_cfg
        self.box_adj = box_adj
        self.borders = borders

        # Загружаем модель SAM2
        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

    def show_mask(self, mask, ax, borders=None):
        """
        Отображение маски на изображении
        """

        if borders is None:
            borders = self.borders

        color = np.array([30/255, 144/255, 255/255, 0.6])

        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)

        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=375):
        """
        Отображение точек на изображении

        Args:
            coords: координаты точек
            labels: метки точек (1 - положительные, 0 - отрицательные)
            ax: ось matplotlib
            marker_size: размер маркера
        """
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]

        if len(pos_points) > 0:
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
                      marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        if len(neg_points) > 0:
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
                      marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def show_box(self, box, ax, color='green', linewidth=2):
        """
        Отображение bounding box на изображении
        """
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color,
                                  facecolor=(0, 0, 0, 0), lw=linewidth))

    def show_masks(self, image, masks, scores, point_coords=None,
                   box_coords=None, input_labels=None, borders=True):
        """
        Отображение нескольких масок на одном изображении
        """
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(4, 4))
            plt.imshow(image)
            self.show_mask(mask, plt.gca(), borders=borders)

            if point_coords is not None:
                assert input_labels is not None
                self.show_points(point_coords, input_labels, plt.gca())
            if box_coords is not None:
                self.show_box(box_coords, plt.gca())
            if len(scores) > 1:
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)

            plt.axis('off')
            plt.show()

    def read_box_from_file(self, box_path):
        """
        Чтение bounding box из файла
        """
        if not os.path.exists(box_path):
            print(f"Файл {box_path} не найден")
            return None

        with open(box_path) as f:
            input_box = list(map(float, f.read().split(',')))

        # Применяем корректировки к box
        input_box[0] += self.box_adj.get('left', 0)
        input_box[1] += self.box_adj.get('top', 0)
        input_box[2] += self.box_adj.get('right', 0)
        input_box[3] += self.box_adj.get('bottom', 0)

        return input_box

    def generate_points_from_box(self, input_box):
        """
        Генерация точек для SAM на основе bounding box
        """
        width = input_box[2] - input_box[0]
        height = input_box[3] - input_box[1]
        c_x = (input_box[0] + input_box[2]) / 2
        c_y = (input_box[1] + input_box[3]) / 2

        input_point = np.array([
            [c_x, c_y],
            [c_x, input_box[1] + 10],
            [c_x + 0.25 * width, c_y + 0.25 * height],
            [c_x - 0.33 * width, c_y - 0.33 * height],
            [c_x + 0.25 * width, c_y - 0.25 * height],
            [c_x - 0.25 * width, c_y + 0.25 * height]
        ])
        input_label = np.array([1, 1, 1, 1, 1, 1])

        return input_point, input_label

    def generate_mask_for_image(self, image, box_path, multimask_output=False):
        """
        Генерация маски для одного изображения
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert('RGB'))
        else:
            image_np = image

        self.predictor.set_image(image_np)

        input_box = self.read_box_from_file(box_path)
        if input_box is None:
            return None, None, None

        input_point, input_label = self.generate_points_from_box(input_box)

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=multimask_output,
        )

        return masks, scores, input_box

    def save_mask(self, mask, save_path):
        """
        Сохранение маски в файл
        """
        Image.fromarray((mask * 255).astype(np.uint8)).save(save_path)

    def process_folders(self, folder_a, folder_b, folder_c,
                        save_masks=True, show_visualization=True):
        """
        Обработка всех изображений в трех папках
        """
        # Получаем списки файлов
        files_a = self._get_image_files(folder_a)
        files_b = self._get_image_files(folder_b)
        files_c = self._get_image_files(folder_c)

        results = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'masks_info': []
        }

        for i in range(min(len(files_a), len(files_b), len(files_c))):
            # Загружаем изображения
            img_a = Image.open(os.path.join(folder_a, files_a[i])).convert('RGB')
            img_b = Image.open(os.path.join(folder_b, files_b[i])).convert('RGB')
            img_c = Image.open(os.path.join(folder_c, files_c[i])).convert('RGB')

            images = [img_a, img_b, img_c]
            folders = [folder_a, folder_b, folder_c]
            titles = ['generated_faces', 'generated_faces_older', 'generated_faces_glasses']
            filenames = [files_a[i], files_b[i], files_c[i]]

            if show_visualization:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f'Маски для изображения {i+1}', fontsize=16)

            for idx, (img, folder, title, filename) in enumerate(zip(images, folders, titles, filenames)):
                box_path = f'{folder}/box_person_{i+1}.txt'

                # Генерируем маску
                masks, scores, input_box = self.generate_mask_for_image(img, box_path)

                if masks is not None and len(masks) > 0:
                    mask = masks[0]
                    score = scores[0]

                    if save_masks:
                        mask_path = f'{folder}/mask_person_{i+1}.jpeg'
                        self.save_mask(mask, mask_path)

                    results['successful'] += 1
                    results['masks_info'].append({
                        'index': i,
                        'folder': folder,
                        'filename': filename,
                        'score': float(score)
                    })

                    if show_visualization:
                        img_np = np.array(img)
                        axes[idx].imshow(img_np)
                        self.show_mask(mask, axes[idx])
                        self.show_box(input_box, axes[idx])
                        axes[idx].set_title(f'{title}\nScore: {score:.3f}')
                        axes[idx].axis('off')
                else:
                    results['failed'] += 1
                    if show_visualization:
                        img_np = np.array(img)
                        axes[idx].imshow(img_np)
                        axes[idx].set_title(f'{title}\n(маска не создана)')
                        axes[idx].axis('off')

            results['total_processed'] += 1

            if show_visualization:
                plt.tight_layout()
                plt.show()
                plt.close(fig)

            # Закрываем изображения
            img_a.close()
            img_b.close()
            img_c.close()

        return results

    def _get_image_files(self, folder_path):
        """
        Получение списка файлов изображений из папки
        """
        return sorted([
            f for f in os.listdir(folder_path)
            if (f.lower().endswith(('.png', '.jpg', '.jpeg')) and
                f.lower().startswith(('person_')))
        ])

