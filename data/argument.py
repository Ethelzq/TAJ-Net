import cv2
from skimage.transform import AffineTransform, warp
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as AF
import skimage

def affine_image(img):
    """

    Args:
        img: (h, w) or (1, h, w)

    Returns:
        img: (h, w)
    """
    # ch, h, w = img.shape
    # img = img / 255.
#    if img.ndim == 3:
#        img = img[0]

    sx = 1
    sy = 1
    rot_angle = 0
    shear_angle = 0
    tx = 0
    ty = 0

    # --- scale ---
    if np.random.rand() < 0.2:
        min_scale = 0.8
        max_scale = 1.2
        sx = np.random.uniform(min_scale, max_scale)
        sy = np.random.uniform(min_scale, max_scale)

    # --- rotation ---
    if np.random.rand() < 0.2:
        max_rot_angle = 15
        rot_angle = np.random.uniform(-max_rot_angle, max_rot_angle) * np.pi / 180.

    # --- shear ---
    if np.random.rand() < 0.2:
        max_shear_angle = 20
        shear_angle = np.random.uniform(-max_shear_angle, max_shear_angle) * np.pi / 180.

    # --- translation ---
    if np.random.rand() < 0.2:
        max_translation = 10
        tx = np.random.randint(-max_translation, max_translation)
        ty = np.random.randint(-max_translation, max_translation)

    tform = AffineTransform(scale=(sx, sy), rotation=rot_angle, shear=shear_angle,
                            translation=(tx, ty))
    transformed_image = warp(img, tform)
   
    return transformed_image

class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.

    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """
    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            #n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                             int(i * grid_h) : int(i * grid_h + grid_h / 2),
                             int(j * grid_w) : int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value
                
                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)
                
    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = AF.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')
    
def crop_char_image(image, threshold=5./255.):
    assert image.ndim == 2
    is_black = image > threshold

    is_black_vertical = np.sum(is_black, axis=0) > 0
    is_black_horizontal = np.sum(is_black, axis=1) > 0
    left = np.argmax(is_black_horizontal)
    right = np.argmax(is_black_horizontal[::-1])
    top = np.argmax(is_black_vertical)
    bottom = np.argmax(is_black_vertical[::-1])
    height, width = image.shape
    cropped_image = image[left:height - right, top:width - bottom]
    # print(left, )
    return cropped_image


def resize(image, size=(128, 128)):
    return cv2.resize(image, size)


def add_gaussian_noise(x, sigma):
    x += np.random.randn(*x.shape) * sigma
    x = np.clip(x, 0., 1.)
    return x


def _evaluate_ratio(ratio):
    if ratio <= 0.:
        return False
    return np.random.uniform() < ratio


def apply_aug(aug, image):
    return aug(image=image)['image']


class Transform:
    def __init__(self, affine=False, size=None,
                 train=True, 
                 sigma=-1.,GridMask_ration=0, blur_ratio=0., noise_ratio=0., cutout_ratio=0.,
                 grid_distortion_ratio=0., elastic_distortion_ratio=0., random_brightness_ratio=0.,
                 piece_affine_ratio=0., ssr_ratio=0.):
        self.affine = affine

        self.size = size

        self.train = train

        self.sigma = sigma / 255.
        self.GridMask_ration = GridMask_ration
        self.blur_ratio = blur_ratio
        self.noise_ratio = noise_ratio
        self.cutout_ratio = cutout_ratio
        self.grid_distortion_ratio = grid_distortion_ratio
        self.elastic_distortion_ratio = elastic_distortion_ratio
        self.random_brightness_ratio = random_brightness_ratio
        self.piece_affine_ratio = piece_affine_ratio
        self.ssr_ratio = ssr_ratio

    def __call__(self, example):
        if self.train:
            x, y = example
        else:
            x = example
        # --- Augmentation ---
        if self.affine:
            x = affine_image(x)

        # # --- Train/Test common preprocessing ---
        # if (self.size is not None) and (x.shape[0:2] != self.size):
        #     x = skimage.transform.resize(x, output_shape=(self.size[1], self.size[2], self.size[0]))
        #     # x = resize(x, size=self.size)
        # if self.sigma > 0.:
        #     x = add_gaussian_noise(x, sigma=self.sigma)
        #
        # # albumentations...
        #
        #
        # if _evaluate_ratio(self.GridMask_ration):
        #     r = np.random.uniform()
        #     if r < 0.25:
        #         x = apply_aug(GridMask(num_grid=3, mode=0,p=1.0), x)
        #     elif r < 0.5:
        #         x = apply_aug(GridMask(num_grid=3, mode=1,p=1.0), x)
        #     elif r < 0.75:
        #         x = apply_aug(GridMask(num_grid=3, mode=2,p=1.0), x)
        #     else:
        #         x = apply_aug(GridMask(num_grid=(3,5), mode=1,p=1.0), x)
        # # 1. blur
        # if _evaluate_ratio(self.blur_ratio):
        #     r = np.random.uniform()
        #     if r < 0.25:
        #         x = apply_aug(A.Blur(p=1.0), x)
        #     elif r < 0.5:
        #         x = apply_aug(A.MedianBlur(blur_limit=5, p=1.0), x)
        #     elif r < 0.75:
        #         x = apply_aug(A.GaussianBlur(p=1.0), x)
        #     else:
        #         x = apply_aug(A.MotionBlur(p=1.0), x)
        #
        # if _evaluate_ratio(self.noise_ratio):
        #     r = np.random.uniform()
        #     if r < 0.50:
        #         x = apply_aug(A.GaussNoise(var_limit=5. / 255., p=1.0), x)
        #     else:
        #         x = apply_aug(A.MultiplicativeNoise(p=1.0), x)
        #
        # if _evaluate_ratio(self.cutout_ratio):
        #     # A.Cutout(num_holes=2,  max_h_size=2, max_w_size=2, p=1.0)  # Deprecated...
        #     x = apply_aug(A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=1.0), x)
        #
        # if _evaluate_ratio(self.grid_distortion_ratio):
        #     x = apply_aug(A.GridDistortion(p=1.0), x)
        #
        # if _evaluate_ratio(self.elastic_distortion_ratio):
        #     x = apply_aug(A.ElasticTransform(
        #         sigma=50, alpha=1, alpha_affine=10, p=1.0), x)
        #
        # if _evaluate_ratio(self.random_brightness_ratio):
        #     # A.RandomBrightness(p=1.0)  # Deprecated...
        #     # A.RandomContrast(p=1.0)    # Deprecated...
        #     x = apply_aug(A.RandomBrightnessContrast(p=1.0), x)
        #
        # if _evaluate_ratio(self.piece_affine_ratio):
        #     x = apply_aug(A.IAAPiecewiseAffine(p=1.0), x)
        #
        # if _evaluate_ratio(self.ssr_ratio):
        #     x = apply_aug(A.ShiftScaleRotate(
        #         shift_limit=0.0625,
        #         scale_limit=0.1,
        #         rotate_limit=30,
        #         p=1.0), x)
        # print(x.shape)
        x = x.astype(np.float32)
        x = np.transpose(x,(2,0,1))
        if self.train:
            y = y.astype(np.int64)
            return x, y
        else:
            return x