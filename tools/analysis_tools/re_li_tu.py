# @Version  :1.0
# @Author   :High_cold
# @File     :re_li_tu.py
# @Time     :2025/9/26 下午8:10
"""
Use the pytorch-grad-cam tool to visualize Class Activation Maps (CAM).
requirement: pip install grad-cam
"""

from argparse import ArgumentParser
import os

import numpy as np
import torch
import torch.nn.functional as F
from mmengine import Config
from mmengine.model import revert_sync_batchnorm
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image

from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmseg.utils import register_all_modules


class SemanticSegmentationTarget:
    """wrap the model.
    requirement: pip install grad-cam
    Args:
        category (int): Visualization class.
        mask (ndarray): Mask of class.
        size (tuple): Image size.
    """

    def __init__(self, category, mask, size):
        self.category = category
        self.mask = torch.from_numpy(mask)
        self.size = size
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        model_output = torch.unsqueeze(model_output, dim=0)
        model_output = F.interpolate(
            model_output, size=self.size, mode='bilinear', align_corners=True)
        model_output = torch.squeeze(model_output, dim=0)

        return (model_output[self.category, :, :] * self.mask).sum()


def main():
    parser = ArgumentParser()

    # 设置预测文件夹路径
    parser.add_argument('--input_dir', default='/home/user/mmsegmentation-main-2018-improve/data/EndoVis2018/JPEGImages', help='Input directory containing images')

    # 设置配置文件路径
    parser.add_argument('--config',
                        default='/home/user/mmsegmentation-main-2018-improve/work_dirs/mobilenet-v2-d8_deeplabv3plus_2xb8-30k_EndoVis2018-512x512-Eis-400-3/mobilenet-v2-d8_deeplabv3plus_2xb8-30k_EndoVis2018-512x512.py',
                        help='Config file')

    # 设置权重文件路径
    parser.add_argument('--checkpoint', default='/home/user/mmsegmentation-main-2018-improve/work_dirs/mobilenet-v2-d8_deeplabv3plus_2xb8-30k_EndoVis2018-512x512-Eis-400-3/best_mIoU_epoch_388.pth',
                        help='Checkpoint file')

    # 设置保存输出的热力图文件夹路径
    parser.add_argument(
        '--out-dir',
        default='/home/user/mmsegmentation-main-2018-improve/work_results_cam/EiS_mv2/11',
        help='Directory to save CAM images')

    # 设置需要输出网络特定层的热力图
    parser.add_argument(
        '--target-layers',
        default='decode_head.conv_seg',
        help='Target layers to visualize CAM')

    # Grad-CAM为类激活加权映射，所以需要设置固定类别的索引来绘制，需要与传入网络训练时候一致
    parser.add_argument(
        '--category-index', default='11', help='Category to visualize CAM')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.out_dir, exist_ok=True)

    # build the model from a config file and a checkpoint file
    register_all_modules()
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # ---------------------- 添加以下代码 ----------------------
    print("Model Structure:")
    # 打印模型中所有模块的名称
    for name, module in model.named_modules():
        print(name)
    # ---------------------------------------------------------
    # 列出输入目录中的所有图片文件
    input_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for input_file in input_files:
        input_path = os.path.join(args.input_dir, input_file)

        # test a single image
        result = inference_model(model, input_path)

        # result data conversion
        prediction_data = result.pred_sem_seg.data
        pre_np_data = prediction_data.cpu().numpy().squeeze(0)

        target_layers = args.target_layers
        target_layers = [eval(f'model.{target_layers}')]

        category = int(args.category_index)
        mask_float = np.float32(pre_np_data == category)

        # data processing
        image = np.array(Image.open(input_path).convert('RGB'))
        height, width = image.shape[0], image.shape[1]
        rgb_img = np.float32(image) / 255
        config = Config.fromfile(args.config)
        image_mean = config.data_preprocessor['mean']
        image_std = config.data_preprocessor['std']
        input_tensor = preprocess_image(
            rgb_img,
            mean=[x / 255 for x in image_mean],
            std=[x / 255 for x in image_std])

        # Grad CAM(Class Activation Maps)

        targets = [
            SemanticSegmentationTarget(category, mask_float, (height, width))
        ]
        cam_file_name = os.path.splitext(input_file)[0] + '_cam.jpg'  # 使用输入图片文件名构造保存文件名
        cam_file_path = os.path.join(args.out_dir, cam_file_name)
        # Can also be LayerCAM, XGradCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM，这里可以根据需要选择不同的特征图生成方式
        with GradCAM(
                model=model,
                target_layers=target_layers,
                #                 use_cuda=torch.cuda.is_available()
        ) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # 保存 CAM 图片
            Image.fromarray(cam_image).save(cam_file_path)


if __name__ == '__main__':
    main()