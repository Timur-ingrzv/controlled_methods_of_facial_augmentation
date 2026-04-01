import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from config import GeneratorConfig


class FaceGenerator:
    """Класс для синтеза аугментированных изображений лиц"""
    def __init__(self, gender_detector):
        CODE_DIR = '/content/interfacegan'
        if CODE_DIR not in sys.path:
            sys.path.insert(0, CODE_DIR)

        model_name = GeneratorConfig.generator_model_name
        latent_space_type = GeneratorConfig.latent_space_type
        self.generator = self.build_generator(model_name)
        self.ATTRS = GeneratorConfig.attrs
        self.boundaries = {}
        self.latent_space_type = latent_space_type
        for i, attr_name in enumerate(self.ATTRS):
            boundary_name = f'{model_name}_{attr_name}'
            if self.generator.gan_type == 'stylegan' and self.latent_space_type == 'W':
                self.boundaries[attr_name] = np.load(f'boundaries/{boundary_name}_w_boundary.npy')
            else:
                self.boundaries[attr_name] = np.load(f'boundaries/{boundary_name}_boundary.npy')
        self.age = GeneratorConfig.age_coef
        self.eyeglasses = GeneratorConfig.eyeglasses_coef
        self.gender = GeneratorConfig.gender_coef
        self.pose = GeneratorConfig.pose_coef
        self.smile = GeneratorConfig.smile_coef

        self.gender_detector = gender_detector

    def synthesize_faces(self, dir_path, num_samples=4, filtered_gender='male'):
        os.makedirs(f'{dir_path}', exist_ok=True)
        os.makedirs(f'{dir_path}/generated_faces', exist_ok=True)
        os.makedirs(f'{dir_path}/generated_faces_older', exist_ok=True)
        os.makedirs(f'{dir_path}/generated_faces_glasses', exist_ok=True)

        cur_samples = 0

        while cur_samples < num_samples:
            noise_seed = np.random.randint(0, 1e3)

            latent_codes = self.sample_codes(self.generator, 4, self.latent_space_type, noise_seed)
            if self.generator.gan_type == 'stylegan' and self.latent_space_type == 'W':
                synthesis_kwargs = {'latent_space_type': 'W'}
            else:
                synthesis_kwargs = {}

            images = self.generator.easy_synthesize(latent_codes, **synthesis_kwargs)['image']

            new_codes = latent_codes.copy()
            new_codes += self.boundaries['age'] * self.age
            new_image_ages = self.generator.easy_synthesize(new_codes, **synthesis_kwargs)['image']

            new_codes = latent_codes.copy()
            new_codes += self.boundaries['eyeglasses'] * self.eyeglasses
            new_image_glasses = self.generator.easy_synthesize(new_codes, **synthesis_kwargs)['image']

            for i in range(4):
                image = Image.fromarray(images[i])
                if filtered_gender != 'both' and self.gender_detector.predict_gender(image) != filtered_gender:
                    continue
                image.save(f'{dir_path}/generated_faces/person_{i+1}.jpeg')

                new_image = Image.fromarray(new_image_ages[i])
                new_image.save(f'{dir_path}/generated_faces_older/person_{i+1}.jpeg')

                new_image = self.generator.easy_synthesize(new_codes, **synthesis_kwargs)['image'][i]
                new_image = Image.fromarray(new_image_glasses[i])
                new_image.save(f'{dir_path}/generated_faces_glasses/person_{i+1}.jpeg')

                cur_samples += 1
                if cur_samples == num_samples:
                    break

    def build_generator(self, model_name):
        """
        Создаем генератор
        """
        from models.model_settings import MODEL_POOL
        from models.pggan_generator import PGGANGenerator
        from models.stylegan_generator import StyleGANGenerator
        gan_type = MODEL_POOL[model_name]['gan_type']
        if gan_type == 'pggan':
            generator = PGGANGenerator(model_name)
        elif gan_type == 'stylegan':
            generator = StyleGANGenerator(model_name)
        return generator

    def sample_codes(self, generator, num, latent_space_type='Z', seed=0):
        """
        Сэмплируем латентные представления
        """
        np.random.seed(seed)
        codes = generator.easy_sample(num)
        if generator.gan_type == 'stylegan' and latent_space_type == 'W':
            codes = torch.from_numpy(codes).type(torch.FloatTensor).to(generator.run_device)
            codes = generator.get_value(generator.model.mapping(codes))
        return codes

    def show_image_pairs(self, folder_a, folder_b, folder_c):
        """
        Отображает первые 3 пары изображений с помощью matplotlib
        """
        files_a = sorted([f for f in os.listdir(folder_a)
                        if (f.lower().endswith(('.png', '.jpg', '.jpeg')) and 
                         f.lower().startswith(('person_')))])[:3]
        files_b = sorted([f for f in os.listdir(folder_b)
                        if (f.lower().endswith(('.png', '.jpg', '.jpeg')) and 
                         f.lower().startswith(('person_')))])[:3]
        files_c = sorted([f for f in os.listdir(folder_c)
                        if (f.lower().endswith(('.png', '.jpg', '.jpeg')) and 
                         f.lower().startswith(('person_')))])[:3]
        print(files_a)
        fig, axes = plt.subplots(3, 3, figsize=(10, 12))
        fig.suptitle('Первые 3 тройки изображений', fontsize=16)

        for i in range(min(3, len(files_a), len(files_b))):
            # Папка gen_images
            img_a = Image.open(os.path.join(folder_a, files_a[i]))
            axes[0, i].imshow(img_a)
            axes[0, i].set_title(f'A{i+1}: {files_a[i]}')
            axes[0, i].axis('off')

            # Папка gen_older
            img_b = Image.open(os.path.join(folder_b, files_b[i]))
            axes[1, i].imshow(img_b)
            axes[1, i].set_title(f'B{i+1}: {files_b[i]}')
            axes[1, i].axis('off')

            # Папка gen_glasses
            img_c = Image.open(os.path.join(folder_c, files_c[i]))
            axes[2, i].imshow(img_c)
            axes[2, i].set_title(f'С{i+1}: {files_b[i]}')
            axes[2, i].axis('off')


        plt.tight_layout()
        plt.show();
        