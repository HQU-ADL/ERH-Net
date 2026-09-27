# @Version  :1.0
# @Author   :High_cold
# @File     :mmseg_ref.py
# @Time     :2025/10/1 上午11:39
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import torch.nn.functional as F
from mmengine import Config
from mmengine.model import revert_sync_batchnorm
import numpy as np

np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from mmseg.apis import init_model
from mmseg.utils import register_all_modules
import cv2
import os
from argparse import ArgumentParser
from timm.utils import AverageMeter
import seaborn as sns

plt.rcParams["font.family"] = "Times New Roman"


def get_activation(feat, backbone_idx=-1):
    def hook(model, inputs, outputs):
        if backbone_idx != -1:
            for _ in range(5 - len(outputs)): outputs.insert(0, None)
            feat.append(outputs[backbone_idx])
        else:
            feat.append(outputs)

    return hook


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def preprocess_image(image, mean, std):
    """Preprocess image similar to MMSegmentation's data preprocessor."""
    image = np.float32(image)
    input_tensor = torch.from_numpy(np.transpose(image, axes=[2, 0, 1])).unsqueeze(0)
    mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    input_tensor = (input_tensor - mean) / std
    return input_tensor


def get_rectangle(data, thresh):
    h, w = data.shape
    all_sum = np.sum(data)
    for i in range(1, min(h // 2, w // 2)):
        selected_area = data[h // 2 - i:h // 2 + 1 + i, w // 2 - i:w // 2 + 1 + i]
        area_sum = np.sum(selected_area)
        if area_sum / all_sum > thresh:
            return i * 2 + 1, (i * 2 + 1) / h * (i * 2 + 1) / w
    return None, None


def heatmap(data, camp='RdYlGn', figsize=(10, 10.75), ax=None, save_path=None):
    plt.figure(figsize=figsize, dpi=40)
    ax = sns.heatmap(data,
                     xticklabels=False,
                     yticklabels=False, cmap=camp,
                     center=0, annot=False, ax=ax, cbar=True, annot_kws={"size": 24}, fmt='.2f')
    plt.tight_layout()
    plt.savefig(save_path)


class mmseg_erf:
    feature, hooks = [], []

    def __init__(self, config_path, checkpoint_path, device, target_layer, input_dir, num_images, save_path) -> None:
        self.device = torch.device(device)
        register_all_modules()
        self.config = Config.fromfile(config_path)
        self.model = init_model(config_path, checkpoint_path, device=self.device)
        if self.device.type == 'cpu':
            self.model = revert_sync_batchnorm(self.model)
        self.model.eval()

        # Enable gradients
        for p in self.model.parameters():
            p.requires_grad_(True)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0, weight_decay=0)
        self.meter = AverageMeter()

        # Get img_size from config
        for pipe in self.config.train_pipeline:
            if pipe['type'] == 'RandomCrop':
                self.img_size = pipe['crop_size']
                break
        else:
            self.img_size = (512, 512)  # default

        # Get mean and std
        self.mean = [x / 255.0 for x in self.config.data_preprocessor['mean']]
        self.std = [x / 255.0 for x in self.config.data_preprocessor['std']]

        # Register hook
        target_module = self.model
        for part in target_layer.split('.'):
            target_module = getattr(target_module, part)
        self.hooks.append(target_module.register_forward_hook(get_activation(self.feature)))

        self.input_dir = input_dir
        self.num_images = num_images
        self.save_path = save_path

    def get_input_grad(self, samples):
        self.optimizer.zero_grad()
        _ = self.model(samples, mode='tensor')
        outputs = self.feature[-1]
        self.feature.clear()
        out_size = outputs.size()
        central_point = F.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
        grad = torch.autograd.grad(central_point, samples)
        grad = grad[0]
        grad = F.relu(grad)
        aggregated = grad.sum((0, 1))
        grad_map = aggregated.cpu().numpy()
        return grad_map

    def process(self):
        input_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for input_file in input_files:
            if self.meter.count >= self.num_images:
                break

            input_path = os.path.join(self.input_dir, input_file)
            img = cv2.imread(input_path)
            if img is None:
                continue
            img, _, _ = letterbox(img, new_shape=self.img_size, auto=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            rgb_img = np.float32(img) / 255.0
            input_tensor = preprocess_image(rgb_img, self.mean, self.std)
            input_tensor = input_tensor.to(self.device)
            input_tensor.requires_grad_(True)

            contribution_scores = self.get_input_grad(input_tensor)

            if np.isnan(np.sum(contribution_scores)):
                print('got NAN, next image')
                continue
            else:
                print(f'{self.meter.count + 1}/{self.num_images} calculate....')
                self.meter.update(contribution_scores)

        # Set figure parameters
        large = 24;
        med = 24;
        small = 24
        params = {'axes.titlesize': large,
                  'legend.fontsize': med,
                  'figure.figsize': (16, 10),
                  'axes.labelsize': med,
                  'xtick.labelsize': med,
                  'ytick.labelsize': med,
                  'figure.titlesize': large}
        plt.rcParams.update(params)
        plt.style.use('default')
        sns.set_style("white")
        # plt.rc('font', **{'family': 'Times New Roman'})
        plt.rcParams['axes.unicode_minus'] = False

        data = self.meter.avg
        print(f'max value:{np.max(data):.3f} min value:{np.min(data):.3f}')

        data = np.log10(data + 1)  # take the logarithm for better readability
        data = data / np.max(data)  # rescale to [0,1]
        print('======================= the high-contribution area ratio =====================')
        for thresh in [0.2, 0.3, 0.5, 0.99]:
            side_length, area_ratio = get_rectangle(data, thresh)
            print('thresh, rectangle side length, area ratio: ', thresh, side_length, area_ratio)
        heatmap(data, save_path=self.save_path)


def main():
    parser = ArgumentParser()
    parser.add_argument('--config',
                        default='/home/user/mmsegmentation-main-2018-improve/work_dirs/mobilenet-v2-d8_deeplabv3plus_2xb8-30k_EndoVis2018-512x512-Eis-400-3/mobilenet-v2-d8_deeplabv3plus_2xb8-30k_EndoVis2018-512x512.py',
                        help='Config file')
    parser.add_argument('--checkpoint',
                        default='/home/user/mmsegmentation-main-2018-improve/work_dirs/mobilenet-v2-d8_deeplabv3plus_2xb8-30k_EndoVis2018-512x512-Eis-400-3/best_mIoU_epoch_388.pth',
                        help='Checkpoint file')
    parser.add_argument('--input_dir',
                        default='/home/user/mmsegmentation-main-2018-improve/data/EndoVis2018/JPEGImages',
                        help='Input directory containing images')
    parser.add_argument('--target_layer',
                        default='decode_head.sep_bottleneck', #decode_head.conv_seg
                        help='Target layer for hook (e.g., decode_head.conv_seg)')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--num_images', type=int, default=3232, help='Number of images to process')
    parser.add_argument('--save_path', default='MSCA-G-L2.png', help='Path to save the heatmap')
    args = parser.parse_args()

    mmseg_erf(args.config, args.checkpoint, args.device, args.target_layer, args.input_dir, args.num_images,
              args.save_path).process()


if __name__ == '__main__':
    main()