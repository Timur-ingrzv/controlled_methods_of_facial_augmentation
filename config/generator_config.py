from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class GeneratorConfig:
    gender_detector_model_name = "prithivMLmods/Realistic-Gender-Classification"

    generator_model_name = 'stylegan_ffhq'
    latent_space_type = 'Z'
    attrs = ['age', 'eyeglasses', 'gender', 'pose', 'smile']

    age_coef = 1.1
    eyeglasses_coef = 1.5
    gender_coef = 0
    pose_coef = 0
    smile_coef = 0