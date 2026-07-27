"""
Video-Level Augmentation Module for DeMamba Training

Key principle: All augmentations are applied consistently to ALL frames of a video.
The same random decisions (flip, blur, etc.) are made ONCE per video and applied to every frame.

This ensures temporal consistency and prevents artifacts from per-frame random augmentation.
"""

import random
from typing import List, Tuple

import numpy as np
import albumentations as A


class VideoAugmentor:
    """
    Apply video-level augmentations consistently across all frames.
    
    All random decisions are made ONCE per video, then the same
    transformation is applied to every frame.
    """
    
    def __init__(self, config: dict):
        """
        Initialize augmentor with configuration.
        
        Args:
            config: Full configuration dictionary
        """
        self.aug_config = config['augmentation']
        
        # Extract parameters
        self.resize_height = self.aug_config['resize']['height']
        self.resize_width = self.aug_config['resize']['width']
        
        self.flip_prob = self.aug_config['horizontal_flip']['prob']
        
        self.jpeg_prob = self.aug_config['jpeg_compression']['prob']
        self.jpeg_quality_range = self.aug_config['jpeg_compression']['quality_range']
        
        self.noise_prob = self.aug_config['gaussian_noise']['prob']
        self.noise_var_limit = self.aug_config['gaussian_noise']['var_limit']
        
        self.blur_prob = self.aug_config['gaussian_blur']['prob']
        self.blur_kernel_range = self.aug_config['gaussian_blur']['kernel_range']
        
        self.grayscale_prob = self.aug_config['grayscale']['prob']
        
        self.normalize_mean = self.aug_config['normalize']['mean']
        self.normalize_std = self.aug_config['normalize']['std']
        self.normalize_max_pixel = self.aug_config['normalize']['max_pixel_value']
    
    def __call__(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Apply video-level augmentations to all frames.
        
        Args:
            frames: List of frames as numpy arrays (H, W, C), RGB format
        
        Returns:
            Augmented frames as numpy array (T, C, H, W)
        """
        # Make all random decisions ONCE for this video
        do_flip = random.random() < self.flip_prob
        do_jpeg = random.random() < self.jpeg_prob
        do_noise = random.random() < self.noise_prob
        do_blur = random.random() < self.blur_prob
        do_grayscale = random.random() < self.grayscale_prob
        
        # Sample fixed parameters for this video
        jpeg_quality = random.randint(
            self.jpeg_quality_range[0],
            self.jpeg_quality_range[1]
        ) if do_jpeg else None
        
        blur_kernel = random.choice(
            range(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2)
        ) if do_blur else None
        
        # Build the transform pipeline for THIS video
        transform = self._build_transform(
            do_flip=do_flip,
            do_jpeg=do_jpeg,
            jpeg_quality=jpeg_quality,
            do_noise=do_noise,
            do_blur=do_blur,
            blur_kernel=blur_kernel,
            do_grayscale=do_grayscale
        )
        
        # Apply SAME transform to all frames
        augmented_frames = []
        for frame in frames:
            augmented = transform(image=frame)['image']
            # Convert HWC to CHW
            augmented_frames.append(augmented.transpose(2, 0, 1))
        
        return np.stack(augmented_frames)  # (T, C, H, W)
    
    def _build_transform(
        self,
        do_flip: bool,
        do_jpeg: bool,
        jpeg_quality: int,
        do_noise: bool,
        do_blur: bool,
        blur_kernel: int,
        do_grayscale: bool
    ) -> A.Compose:
        """
        Build albumentations transform pipeline with fixed parameters.
        
        All transforms use p=1.0 because decisions are already made.
        """
        transforms = []
        
        # Always resize first
        transforms.append(A.Resize(self.resize_height, self.resize_width))
        
        # Horizontal flip
        if do_flip:
            transforms.append(A.HorizontalFlip(p=1.0))
        
        # JPEG compression
        if do_jpeg and jpeg_quality is not None:
            transforms.append(A.ImageCompression(
                quality_range=(jpeg_quality, jpeg_quality),
                p=1.0
            ))
        
        # Gaussian noise
        if do_noise:
            # Convert variance to std deviation (sqrt of variance)
            std_min = (self.noise_var_limit[0] ** 0.5) / 255.0
            std_max = (self.noise_var_limit[1] ** 0.5) / 255.0
            transforms.append(A.GaussNoise(
                std_range=(std_min, std_max),
                p=1.0
            ))
        
        # Gaussian blur
        if do_blur and blur_kernel is not None:
            transforms.append(A.GaussianBlur(
                blur_limit=(blur_kernel, blur_kernel),
                p=1.0
            ))
        
        # Grayscale conversion
        if do_grayscale:
            transforms.append(A.ToGray(p=1.0))
        
        # Always normalize at the end
        transforms.append(A.Normalize(
            mean=self.normalize_mean,
            std=self.normalize_std,
            max_pixel_value=self.normalize_max_pixel,
            p=1.0
        ))
        
        return A.Compose(transforms)


class ValidationTransform:
    """
    Validation/Test transform - only resize and normalize, no augmentation.
    """
    
    def __init__(self, config: dict):
        """
        Initialize validation transform.
        
        Args:
            config: Full configuration dictionary
        """
        self.aug_config = config['augmentation']
        
        self.resize_height = self.aug_config['resize']['height']
        self.resize_width = self.aug_config['resize']['width']
        self.normalize_mean = self.aug_config['normalize']['mean']
        self.normalize_std = self.aug_config['normalize']['std']
        self.normalize_max_pixel = self.aug_config['normalize']['max_pixel_value']
        
        self.transform = A.Compose([
            A.Resize(self.resize_height, self.resize_width),
            A.Normalize(
                mean=self.normalize_mean,
                std=self.normalize_std,
                max_pixel_value=self.normalize_max_pixel,
                p=1.0
            )
        ])
    
    def __call__(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Apply validation transforms to all frames.
        
        Args:
            frames: List of frames as numpy arrays (H, W, C), RGB format
        
        Returns:
            Transformed frames as numpy array (T, C, H, W)
        """
        transformed_frames = []
        for frame in frames:
            augmented = self.transform(image=frame)['image']
            transformed_frames.append(augmented.transpose(2, 0, 1))
        
        return np.stack(transformed_frames)


def create_augmentor(config: dict, is_training: bool = True):
    """
    Factory function to create appropriate augmentor.
    
    Args:
        config: Configuration dictionary
        is_training: Whether to create training augmentor or validation transform
    
    Returns:
        VideoAugmentor for training, ValidationTransform for validation
    """
    if is_training:
        return VideoAugmentor(config)
    else:
        return ValidationTransform(config)
