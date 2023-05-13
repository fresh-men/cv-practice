import datetime
import os
from time import time
from turtle import color

import cv2
import skimage

import torch
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import util

from superglue.matching import Matching
from superglue.utils import frame2tensor, make_matching_plot_fast
from skimage import transform
from skimage import measure
from icecream import ic

class Register(object):
    def __init__(self, ve_model_path, ve_model_type, device='cpu', ve_gan_batch_size=1, logger=None, do_debug=False):
        torch.set_grad_enabled(False)
        self.device = device
        self.logger = logger
        self.do_degug = do_debug

    def set_img(self, dataGroup):
        imgs = []
        masks = []
        Is = []
        Imasks = []
        scales = []
        time = datetime.datetime.now()
        if(dataGroup.img_paths):
            for img_path in dataGroup.img_paths:
                img = util.imread(img_path)
                imgs.append(img)
                Is.append(img)
        elif(dataGroup.input_imgs):
            for input_img in dataGroup.input_imgs:
                img = input_img
                imgs.append(img)
                Is.append(img)
        for i, I in enumerate(Is):
            if len(I.shape) > 2:
                I = I.astype('uint8')
                I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            I = util.img_resize(I, dataGroup.fit_size)
            Is[i] = I
            Imask = util.getMask(I)
            Imasks.append(Imask)
            imgs[i] = imgs[i].astype('uint8')
            img = imgs[i] if imgs[i].shape.__len__() == 2 or imgs[i].shape[2] == 1 else cv2.cvtColor(imgs[i],
                                                                                                   cv2.COLOR_BGR2GRAY)
            mask = np.where(img > 2, 1, 0)
            masks.append(mask)
            scales.append(imgs[i].shape[0] / dataGroup.fit_size)


class SuperglueReg(Register):
    def __init__(self, ve_model_path, ve_model_type, superpoint_model_path, superglue_model_path, logger=None,
                 device='cpu', ve_gan_batch_size=1, nms_radius=4, keypoint_threshold=0, max_keypoints=250,
                 sinkhorn_iterations=20, match_threshold=0.02, do_debug=False):
        super(SuperglueReg, self).__init__(ve_model_path=ve_model_path, ve_model_type=ve_model_type, device=device,
                                           ve_gan_batch_size=ve_gan_batch_size, logger=logger, do_debug=do_debug)
        config = {
            'superpoint': {
                'nms_radius': nms_radius,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints,
                'superpoint_model_path': superpoint_model_path
            },
            'superglue': {
                'weights': superglue_model_path,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }
        self.keys = ['keypoints', 'scores', 'descriptors', 'image']
        self.matching = Matching(config).eval().to(device)
        ic('SuperglueReg init')

    def point_detect_all(self, Is):
        time = datetime.datetime.now()
        detected_datas = []
        for i, img in enumerate(Is):
            # Imask = dataGroup.Imasks[i]
            # cv2.imshow('img',img)
            # cv2.imshow('mask',mask)
            # cv2.waitKey()
            img_tensor = frame2tensor(img, self.device)
            last_data = self.matching.superpoint({'image': img_tensor})
            # last_data = util.remove_boundary_point(Imask, last_data)
            last_data['image'] = img_tensor
            # last_data['mask'] = Imask
            detected_datas.append(last_data)
        return detected_datas

    def match(self, data0, data1):
        data0 = {k + '0': data0[k] for k in self.keys}
        data1 = {k + '1': data1[k] for k in self.keys}
        # 非极大抑制
        data0, data1 = util.nms_scale(data0, data1, scale=25)
        
        kpts0 = data0['keypoints0'][0].cpu().numpy()
        kpts1 = data1['keypoints1'][0].cpu().numpy()
        # with torch.no_grad:
        pred = self.matching({**data0, **data1})
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        max_conf = np.max(confidence) * 0.4
        # valid = np.argwhere(matches > -1)
        valid = np.logical_and(matches > -1, confidence > max_conf)
        if (sum(valid) < 4):
            return np.array([]), np.array([]), np.array([])
        kmc = np.transpose(
            np.append(np.append(np.transpose(kpts0, (1, 0)), matches), confidence).reshape(4, (kpts0.shape[0])), (1, 0))
        kmc = kmc[valid]
        # kmc = list(kmc)
        # kmc = sorted(kmc, key=functools.cmp_to_key(self.cmp))
        # kmc = np.array(kmc[:len(kmc)//3])
        # if (sum(valid) > 9):
        #     mask = LPM_filter(kmc[:, 0:2], kpts1[kmc[:, 2].astype(np.uint8).reshape(-1)])
        #     if (sum(mask) > 4):
        #         kmc = kmc[mask]
        # return valid
        mkpts0 = kmc[:, 0:2].astype(np.float32)
        mkpts1 = kpts1[kmc[:, 2].astype(np.uint8).reshape(-1)].astype(np.float32)
        conf = kmc[:, 3]
        return mkpts0, mkpts1, conf

    def calc_affine(self, data0, data1, mkpts0, mkpts1, conf):
        tform, inliners = skimage.measure.ransac((mkpts0, mkpts1), skimage.transform.AffineTransform, min_samples=5,
                                                 residual_threshold=0.25, max_trials=1000, stop_probability=0.99)
        mkpts0 = mkpts0[inliners]
        mkpts1 = mkpts1[inliners]
        color = cm.jet(conf.reshape(-1))
        img0 = (data0['image'][0][0].cpu().numpy() * 255).astype(np.uint8)
        img1 = (data1['image'][0][0].cpu().numpy() * 255).astype(np.uint8)
        if self.do_degug:
            out = make_matching_plot_fast(
                img0, img1, None, None, mkpts0, mkpts1, color, [],
                path=None, show_keypoints=False, opencv_display=True, small_text=[])

    def show_matches(self, data0, data1, mkpts0, mkpts1, conf):
        color = cm.jet(conf.reshape(-1))
        kpts0 = data0['keypoints'][0].cpu().numpy()
        img0 = (data0['image'][0][0].cpu().numpy() * 255).astype(np.uint8)
        kpts1 = data1['keypoints'][0].cpu().numpy()
        img1 = (data1['image'][0][0].cpu().numpy() * 255).astype(np.uint8)
        # text = [
        #     'SuperGlue',
        #     'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        #     'Matches: {}'.format(len(mkpts0))
        # ]
        # k_thresh = self.matching.superpoint.config['keypoint_threshold']
        # m_thresh = self.matching.superglue.config['match_threshold']
        # small_text = [
        #     'Keypoint Threshold: {:.4f}'.format(k_thresh),
        #     'Match Threshold: {:.2f}'.format(m_thresh),
        #     'Image Pair: {:06}:{:06}'.format(0, 1),
        # ]
        out = make_matching_plot_fast(
            img0, img1, kpts0, kpts1, mkpts0, mkpts1, color, text=[],
            path=None, show_keypoints=True, opencv_display=True, small_text=[])

    # def match_all(self):
    # max_match, id_i, id_j = -1, -1, -1
    # match_matrix = []
    # for i in range(len(self.detected_datas)):
    #     matches = []
    #     for j in range(i + 1, len(self.detected_datas)):
    #         mkpts0, mkpts1, conf = self.match(self.detected_datas[i], self.detected_datas[j])
    #         match = {
    #             'mkpts0': mkpts0,
    #             'mkpts1': mkpts1,
    #             'conf': conf,
    #         }
    #         matches.append(match)
    #         if mkpts0.shape[0] > max_match:
    #             max_match = max_match
    #             id_i, id_j = i, j
    #     match_matrix.append(matches)

    def match_with_master_new(self, id, imgs, scales, detected_datas, do_add_boundary=True):
        time_st = datetime.datetime.now()
        # self.logger.info('match_resize_before,master=' + os.path.basename(dataGroup.img_paths[id]))
        tforms = []
        # imgs = dataGroup.imgs
        # scales = dataGroup.scales
        # detected_datas = dataGroup.detected_datas
        self.do_degug = ic.enabled
        # img0 = self.imgs[id]
        for i in range(len(detected_datas)):
            if i == id:
                tforms.append(skimage.transform.AffineTransform())
                continue
            mkpts0, mkpts1, conf = self.match(detected_datas[id], detected_datas[i])
            # if self.do_degug:
            #     self.show_matches(detected_datas[id], detected_datas[i], mkpts0, mkpts1, conf)
            # if mkpts0.shape[0] < 6:
            #     # self.logger.info(os.path.basename(dataGroup.img_paths[i]) + '匹配点数不足')
            #     tforms.append(None)
            #     continue
            img1_shape = imgs[i].shape
            mkpts0 *= scales[id]
            mkpts1 *= scales[i]
            try:
                tform, inliners = skimage.measure.ransac((mkpts0, mkpts1), skimage.transform.AffineTransform, min_samples=4,
                                                        residual_threshold=14, max_trials=1000, stop_probability=0.5)
            except:
                return mkpts0, mkpts1, conf, np.zeros(len(mkpts0))==0
            # # tform, inliners = skimage.measure.ransac((mkpts0, mkpts1), skimage.transform.ProjectiveTransform, min_samples=4,
            # #                                          residual_threshold=2, max_trials=1000, stop_probability=0.99)
            # # tform, inliners = skimage.measure.ransac((mkpts0, mkpts1), skimage.transform.PolynomialTransform, min_samples=4,
            # #                                          residual_threshold=2, max_trials=1000, stop_probability=0.99)
            if self.do_degug:
                # self.show_matches(detected_datas[id], detected_datas[i], mkpts0 / scales[id], mkpts1 / scales[i], conf)
                self.show_matches(detected_datas[id], detected_datas[i], mkpts0[inliners] / scales[id],
                                  mkpts1[inliners] / scales[i], conf)
            
            return mkpts0, mkpts1, conf, inliners

        if (tforms.count(None) == tforms.__len__()):
            return False

        if do_add_boundary:
            left_up_border, right_down_border = util.getBorders_with_ski(imgs, tforms, (0, 0))
        else:
            left_up_border = np.array((0, 0))
            right_down_border = np.array(imgs[id].shape[:2])
        assert np.min(left_up_border) >= 0 and np.min(right_down_border) >= 0

        mx = max(imgs[id].shape[0], imgs[id].shape[1]) * 4
        if np.max(right_down_border) > mx:
            # self.logger.warning('矩阵明显有误,且存在程序bug导致的检测遗漏,border=' + right_down_border.__str__())
            return False
        return True
        # cv2.imshow('img',self.imgs[id])
        # self.imgs[id] = util.addBorder_cv(self.imgs[id], right_down_border, left_up_border)
        # cv2.imshow('img_',self.imgs[id])
        # cv2.imshow('mask',self.masks[id]*255)
        # self.masks[id] = util.addBorder_cv(self.masks[id], right_down_border, left_up_border, cval=1.)
        # cv2.imshow('mask_',self.masks[id]*255)
        # cv2.waitKey()

    def cmp(self, a, b):
        a = a[3]
        b = b[3]
        if a > b:
            return -1
        elif a < b:
            return 1
        else:
            return 0

if __name__ == '__main__':
    img_dir = r'D:\Images\testImgs'
    # img_dir = r'C:\Users\yjoker\AOneDrive\Dataset\配准\ffaandffa'
    # img_dir = r'C:\Users\yjoker\AOneDrive\Dataset\配准\Retinal-Images'
    imgs = []
    img_paths = []
    for i, img_name in enumerate(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        img = util.imread(img_path)
        img_paths.append(img_path)
        imgs.append(util.img_resize(img, 512))
        if i == 1:
            break
    # vegan = VeGan(device='cuda:0', batch_size=2)
    # vegan.convert_img(imgs, img_paths)
    surperglue = SuperglueReg()
    surperglue.set_img(img_paths)
    datas = surperglue.point_detect_all()
    # surperglue.show_point_all()
    mkpts0, mkpts1, conf = surperglue.match(datas[0], datas[1])
    # surperglue.show_matches(datas[0], datas[1], mkpts0, mkpts1, conf)
    # surperglue.calc_affine(datas[0], datas[1], mkpts0, mkpts1, conf)
    # surperglue.match_with_master(0)
    # surperglue.registering(img_paths,tar_size=512,write_path=r'C:\Users\yjoker\AOneDrive\Dataset\配准\out\test')
#
