# @Version  :1.0
# @Author   :High_cold
# @File     :test.py
# @Time     :2025/8/25 上午11:42
"""
@title: 根据配置(Config)文件和已训练好的参数(pth)进行推理
@Date: 2023/09/14
"""
# 导入必要的库
import os
import numpy as np
import cv2
from tqdm import tqdm
from mmseg.apis import init_model, inference_model, show_result_pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# from mmseg.datasets import CAGDataset
import numpy as np
import mmcv
from PIL import Image


def predict_single_img(img_path, model, save=False, save_path=None, save_model=1, show=False, show_model=1):
    # 读取图像
    img_bgr = cv2.imread(img_path)

    # 语义分割预测
    result = inference_model(model, img_bgr)
    pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
    # print(pred_mask.shape)
    # print(np.unique(pred_mask))

    # 将预测的整数ID, 映射为对应类别的颜色
    pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
    for idx in palette_dict.keys():
        pred_mask_bgr[np.where(pred_mask == idx)] = palette_dict[idx]
    pred_mask_bgr = pred_mask_bgr.astype('uint8')

    # 保存分割结果
    if save:
        # 保存方式一: 直接保存\显示分割(预测)结果
        if save_model == 1:
            save_image_path = os.path.join(save_path, 'pred-' + img_path.split('\\')[-1])
            cv2.imwrite(save_image_path, pred_mask_bgr)
        # 保存方式二: 将分割(预测)结果和原图叠加保存
        elif save_model == 2:
            opacity = 0.5  # 透明度,取值范围(0,1),越大越接近原图
            pred_add = cv2.addWeighted(img_bgr, opacity, pred_mask_bgr, 1 - opacity, 0)
            save_image_path = os.path.join(save_path, 'pred-' + img_path.split('/')[-1])
            cv2.imwrite(save_image_path, pred_add)
        # 保存方式三:检测分割结果的边界轮廓,然后叠加到原图上
        elif save_model == 3:
            # 预测图转为灰度图
            binary = np.where(0.5 < pred_mask, 1, 0).astype(dtype=np.uint8)
            binary = binary * 255
            # 检测mask的边界
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 绘制边界轮廓
            # cv2.drawContours(image, contours, -1, (255, 0, 0), 1)    # 蓝色    # (255, 0, 0) 蓝色； (0, 255, 0)绿色； (0, 0, 255)蓝色
            cv2.drawContours(img_bgr, contours, -1, (0, 255, 0), 1)  # 绿色
            # 保存
            # save_image_path = os.path.join(save_path, 'pred-' + img_path.split('\\')[-1])
            save_image_path = os.path.join(save_path, img_path.split('\\')[-1])
            cv2.imwrite(save_image_path, img_bgr)
            # cv2.imencode(".png", img_bgr)[1].tofile(save_image_path)

    # 显示分割结果
    if show:
        # #****** 可视化语义分割预测结果——方法一（直接显示分割结果）******#
        if show_model == 1:
            plt.figure(figsize=(8, 8))
            plt.imshow(pred_mask)
            plt.savefig('test_result/k1-0.jpg')
            plt.show()
            print(result.seg_logits.data.shape)  # 定量

        # #****** 可视化语义分割预测结果--方法二（叠加在原因上进行显示）******#
        elif show_model == 2:
            # 显示语义分割结果
            plt.figure(figsize=(10, 8))
            plt.imshow(img_bgr[:, :, ::-1])
            plt.imshow(pred_mask, alpha=0.55)
            plt.axis('off')
            plt.savefig('test_result/k1-1.jpg')
            plt.show()

        # #****** 可视化语义分割预测结果--方法三(和原图并排显示) ******#
        elif show_model == 3:
            plt.figure(figsize=(14, 8))
            plt.subplot(1, 2, 1)
            plt.imshow(img_bgr[:, :, ::-1])
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(img_bgr[:, :, ::-1])
            plt.imshow(pred_mask, alpha=0.6)
            plt.axis('off')
            plt.savefig('test_result/k1-2.jpg')
            plt.show()

        # #****** 可视化语义分割预测结果-方法四(按配色方案叠加在原图上显示) ******#
        elif show_model == 4:
            opacity = 0.3  # 透明度,越大越接近原图

            # 将预测的整数ID,映射为对应类别的颜色
            pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
            for idx in palette_dict.keys():
                pred_mask_bgr[np.where(pred_mask == idx)] = palette_dict[idx]
            pred_mask_bgr = pred_mask_bgr.astype('uint8')

            # 将语义分割预测图和原图叠加显示
            pred_viz = cv2.addWeighted(img_bgr, opacity, pred_mask_bgr, 1 - opacity, 0)

            cv2.imwrite('test_result/k1-3.jpg', pred_viz)

            plt.figure(figsize=(8, 8))
            plt.imshow(pred_viz[:, :, ::-1])
            plt.show()

        # #***** 可视化语义分割预测结果-方法五(按mmseg/datasets/cag.py里定义的类别颜色可视化) ***** #
        elif show_model == 5:
            img_viz = show_result_pyplot(model, img_path, result, opacity=0.8, title='MMSeg',
                                         out_file='test_result/k1-4.jpg')
            print('the shape of img_viz:', img_viz.shape)
            plt.figure(figsize=(14, 8))
            plt.imshow(img_viz)
            plt.show()

        # #***** 可视化语义分割预测结果--方法六(加图例)  ***** #
        elif show_model == 6:
            # 获取类别名和调色板
            classes = CAGDataset.METAINFO['classes']
            palette = CAGDataset.METAINFO['palette']
            opacity = 0.15

            # 将分割图按调色板染色
            seg_map = pred_mask.astype('uint8')
            seg_img = Image.fromarray(seg_map).convert('P')
            seg_img.putpalette(np.array(palette, dtype=np.uint8))

            # from matplotlib import pyplot as plt
            # import matplotlib.patches as mpatches
            plt.figure(figsize=(14, 8))
            img_plot = ((np.array(seg_img.convert('RGB'))) * (1 - opacity) + mmcv.imread(img_path) * opacity) / 255
            im = plt.imshow(img_plot)

            # 为每一种颜色创建一个图例
            patches = [mpatches.Patch(color=np.array(palette[i]) / 255, label=classes[i]) for i in range(len(classes))]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')

            plt.savefig('test_result/k1-6.jpg')
            plt.show()


if __name__ == '__main__':
    # 指定测试集路径
    test_images_path = 'data/EndoVis2018/JPEGImages'
    # 指定测试集结果存放路径
    result_save_path = 'work_results/2018-improve-r50'

    # 检测结果存放的目录是否存在.不存在的话,创建空文件夹
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)

    # 载入模型
    # 模型config配置文件
    config_file = 'work_dirs/deeplabv3plus_r50-d8_2xb8-30k_EndoVis2018-512x512-400-2/deeplabv3plus_r50-d8_2xb8-30k_EndoVis2018-512x512.py'
    # 模型checkpoint权重文件
    checkpoint_file = 'work_dirs/deeplabv3plus_r50-d8_2xb8-30k_EndoVis2018-512x512-400-2/best_mIoU_epoch_204.pth'

    # 计算硬件
    device = 'cuda:0'

    # 指定各个类别的BGR配色
    # classes = ('Background', 'Shaft', 'Wrist', 'Claspers', 'Probe'),
    # palette = [[0, 0, 0], [0, 255, 0], [0, 255, 255], [125, 255, 12],
    #            [255, 55, 0]])
    # METAINFO = dict(
    #     classes=('background-tissue', 'instrument-shaft', 'instrument-clasper', 'instrument-wrist', 'kidney-parenchyma',
    #              'covered-kidney', 'thread', 'clamps', 'suturing-needle', 'suction-instrument', 'small-intestine',
    #              'ultrasound-probe'),
    #     palette=[[0, 0, 0], [0, 255, 0], [0, 255, 255], [125, 255, 12],
    #              [255, 55, 0], [24, 55, 125], [187, 155, 25], [0, 255, 125],
    #              [255, 255, 125], [123, 15, 175], [124, 155, 5], [12, 255, 141]])

    palette = [
        # ['background-tissue', [0, 0, 0]],
        # ['instrument-shaft', [0, 255, 0]],
        # ['instrument-clasper', [0, 255, 255]],
        # ['instrument-wrist', [125, 255, 12]],
        # ['kidney-parenchyma', [255, 55, 0]],
        # ['covered-kidney', [24, 55, 125]],
        # ['thread', [187, 155, 25]],
        # ['clamps', [0, 255, 125]],
        # ['suturing-needle', [255, 255, 125]],
        # ['suction-instrument', [123, 15, 175]],
        # ['small-intestine', [124, 155, 5]],
        # ['ultrasound-probe', [12, 255, 141]]

        ['background-tissue', [0, 0, 0]],
        ['instrument-shaft', [0, 255, 0]],
        ['instrument-clasper', [255, 255, 0]],
        ['instrument-wrist', [12, 255, 125]],
        ['kidney-parenchyma', [0, 55, 255]],
        ['covered-kidney', [125, 55, 24]],
        ['thread', [25, 155, 187]],
        ['clamps', [125, 255, 0]],
        ['suturing-needle', [125, 255, 255]],
        ['suction-instrument', [175, 15, 123]],
        ['small-intestine', [5, 155, 124]],
        ['ultrasound-probe', [141, 255, 12]]

    ]
    palette_dict = {}
    for idx, each in enumerate(palette):
        palette_dict[idx] = each[1]
    print(palette_dict)

    # 加载模型
    model = init_model(config_file, checkpoint_file, device=device)

    # 单张图像预测函数
    opacity = 0.7

    # 测试集批量预测
    for each in tqdm(os.listdir(test_images_path)):
        print(each)
        image_path = os.path.join(test_images_path, each)
        # predict_single_img(image_path, model, save=True, save_path=result_save_path, save_model=3, show=False)
        predict_single_img(image_path, model, save=True, save_path=result_save_path, save_model=2, show=False,
                           show_model=2)