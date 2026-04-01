import cv2
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import torch
from IPython import display
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from config import DetectorConfig
plt.rcParams.update({'figure.figsize': (6, 6)})

class FaceDetector:
    """Класс для детекции лиц на изображениях и визуализации результатов"""

    def __init__(self, margin=DetectorConfig.margin):
        """
        Инициализация детектора лиц
        """
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(keep_all=True, margin=margin)
        self.box_expand_left = DetectorConfig.box_expand_left
        self.box_expand_right = DetectorConfig.box_expand_right
        self.box_expand_top = DetectorConfig.box_expand_top
        self.box_expand_bottom = DetectorConfig.box_expand_bottom
    
    def expand_bbox(self, box):
        """
        Расширяет bounding box для охвата полной области лица
        """
        expanded_box = box.copy()
        expanded_box[0] -= self.box_expand_left
        expanded_box[2] += self.box_expand_right
        expanded_box[1] -= self.box_expand_top
        expanded_box[3] += self.box_expand_bottom
        return expanded_box
    
    def draw_bbox(self, image, box, color=(255, 0, 0), width=6):
        """
        Рисует bounding box на изображении
        """
        image_draw = image.copy()
        draw = ImageDraw.Draw(image_draw)
        draw.rectangle(box.tolist(), outline=color, width=width)
        return image_draw
    
    def detect(self, folder_a, folder_b, folder_c, visualise=True, num_images=3):
        """
        Детекция лиц
        """
        files_a = self._get_image_files(folder_a)
        files_b = self._get_image_files(folder_b)
        files_c = self._get_image_files(folder_c)
        
        results = {
            'detections': [],
            'total_images': len(files_a),
            'faces_found': 0,
            'faces_not_found': 0
        }
        
        for i in range(min(len(files_a), len(files_b), len(files_c))):
            img_a = Image.open(os.path.join(folder_a, files_a[i]))
            img_b = Image.open(os.path.join(folder_b, files_b[i]))
            img_c = Image.open(os.path.join(folder_c, files_c[i]))
            
            images = [img_a, img_b, img_c]
            folders = [folder_a, folder_b, folder_c]
            titles = ['generated_faces', 'generated_faces_older', 'generated_faces_glasses']
            filenames = [files_a[i], files_b[i], files_c[i]]
            
            if visualise and i < num_images:
                plt.ioff()
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f'Изображение {i+1}', fontsize=16)
            
            detection_result = {'index': i, 'faces': []}
            
            for idx, (frame, folder, title, filename) in enumerate(zip(images, folders, titles, filenames)):
                boxes, _ = self.mtcnn.detect(frame)
                
                if boxes is not None and len(boxes) > 0:
                    box = boxes[0].copy()
                    
                    box = self.expand_bbox(box)
                    
                    box_path = f'{folder}/box_person_{i+1}.txt'
                    with open(box_path, 'w') as f:
                        f.write(','.join(map(str, box)))
                    
                    frame_draw = self.draw_bbox(frame, box)
                    
                    detection_result['faces'].append({
                        'index': idx,
                        'found': True,
                        'box': box,
                        'folder': folder,
                        'filename': filename
                    })
                    results['faces_found'] += 1
                    
                    if visualise and i < num_images:
                        axes[idx].imshow(frame_draw)
                        axes[idx].set_title(f'{title}\n{filename}')
                        axes[idx].axis('off')
                else:
                    results['faces_not_found'] += 1
            
            results['detections'].append(detection_result)
            
            if visualise:
                plt.tight_layout()
                plt.show();
            
            img_a.close()
            img_b.close()
            img_c.close()
        
        return results
    
    def _get_image_files(self, folder_path):
        """
        Получает отсортированный список файлов изображений из папки
        """
        return sorted([
            f for f in os.listdir(folder_path)
            if (f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')) and
                f.lower().startswith(('person_')))
        ])
    
    def detect_single_image(self, image_path, save_box_path=None, draw_bbox=True):
        """
        Детекция лица на одном изображении (нужно для детекции на шаблоне)
        """
        image = Image.open(image_path)
        boxes, _ = self.mtcnn.detect(image)
        
        box = boxes[0].copy()
        #box = self.expand_bbox(box)
            
        if save_box_path:
            self.save_bbox_coordinates(box, save_box_path)
            
        return image, box, True

    