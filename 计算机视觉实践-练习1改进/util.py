from time import time
from turtle import color
import cv2
import numpy as np
import skimage
import torch
from matplotlib import pyplot as plt
from icecream import ic
def imread(image_path, flags=cv2.IMREAD_UNCHANGED):
    a = np.fromfile(image_path, dtype=np.uint8)
    original_image = cv2.imdecode(a, flags)
    return original_image


def imwrite(path, img, suffix='.bmp'):
    cv2.imencode(suffix, img)[1].tofile(path)


def img_resize(img, tar_size):
    """
    图片小边缩放为指定大小(改成了把长缩放到指定大小）
    :param img:
    :param tar_size: 如512
    :return:
    """
    # h, w = img.shape[0:2]
    # s = 1.0 * tar_size / min(h, w)
    # return cv2.resize(img, (0, 0), fx=s, fy=s)
    if tar_size==-1:
        return img
    s = 1.0 * tar_size / img.shape[0]
    return cv2.resize(img, (0, 0), fx=s, fy=s)

def calc_scale_with_resize(img,mask, fit_size, tar_size):
    if tar_size != -1:
        img=img_resize(img,tar_size)
        mask=img_resize(mask,tar_size)
        scale = tar_size / fit_size
    else:
        h, w = img.shape[0:2]
        scale = min(h, w) / fit_size
    return img,mask,scale

def calc_scale(img, fit_size, tar_size):
    if tar_size != -1:
        scale = tar_size / fit_size
    else:
        h, w = img.shape[0:2]
        scale = min(h, w) / fit_size
    return scale

def calc_from(img,tar_size):
    if tar_size == -1:
        return skimage.transform.AffineTransform()
    else:
        h, w = img.shape[0:2]
        scale = min(h, w)/tar_size
        return skimage.transform.AffineTransform(scale=(scale,scale))

def remove_boundary_point(mask, last_data):
    kps = last_data['keypoints'][0].cpu().numpy().astype(np.int32)
    # if mask==None:
    #     mask = np.where(img <= 3, 1, 0)
    #     mask = np.array(mask, dtype=np.uint8)
    #     conv = np.ones((7, 7), dtype=np.uint8)
    # mask = scipy.signal.convolve(mask, conv, 'same', 'direct')
    valid = np.argwhere(mask[kps[:, 1], kps[:, 0]] == 0).reshape(-1)
    keypoints = [last_data['keypoints'][0][valid, :]]
    descriptors = [last_data['descriptors'][0][:, valid]]
    scores = [last_data['scores'][0][valid]]
    return {
        'keypoints': keypoints,
        'scores': scores,
        'descriptors': descriptors,
    }


def check_transform(shape, tform):
    if np.abs(tform.shear)>0.1:
        return False
    if np.max(tform.scale)/np.min(tform.scale)> 1.5:
        return False
    mx = max(shape[0], shape[1]) * 2
    value = np.array([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    value = tform.inverse(value)
    if np.max(value) > mx or np.min(value) < -mx or np.isnan(np.min(value)):
        return False

    return True


def try_remove(img, mask):
    rows, cols = img.shape[0], img.shape[1]
    src_cols = np.linspace(0, cols - 1, 20).astype(np.int32)
    src_rows = np.linspace(0, rows - 1, 10).astype(np.int32)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    kps = np.dstack([src_cols.flat, src_rows.flat])[0]
    cv2.imshow('before', draw_keypoints(img, kps))
    valid = np.argwhere(mask[kps[:, 0], kps[:, 1]] == 0).reshape(-1)
    kps = kps[valid]
    cv2.imshow('after', draw_keypoints(img, kps))
    cv2.waitKey()


def getMask(img, threshold=3, confine=5, iterations=1):
    """
    获取图像黑边，用来去边缘点或者算重叠面积
    :param img:
    :return:  边缘处mask==1
    """
    mask = np.where(img <= threshold, 1, 0)
    # cv2.imshow('maskbefore',mask.astype(np.uint8)*255)
    mask = mask.astype(np.uint8)
    kernel = np.ones((confine, confine), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=iterations)
    # cv2.imshow('maskafter',mask.astype(np.uint8)*255)
    # cv2.waitKey()
    return mask


def draw_keypoints(img, kp):
    '''
        arguments: gray images
                kp1 is shape Nx2, N number of feature points, first point in horizontal direction
                '''
    image = np.copy(img)
    for i in range(kp.shape[0]):
        image = cv2.circle(image, (np.int32(kp[i, 0]), np.int32(kp[i, 1])), 2, (255, 0, 0), thickness=-1)
    return image


def getLeftUpBorder_with_ski(tform):
    x = np.array([[0, 0]])
    return tform.inverse(x).astype(np.int_)


def getLeftUpBorders_with_ski(tforms):
    x = np.array([[0, 0]])
    z = np.array([0, 0])
    for tform in tforms:
        if tform is None:
            continue
        y = tform.inverse(x)[0]
        z = np.where(y > z, y, z)
    return z


def getBorders_with_ski(imgs, tforms, shape):
    """

    Args:
        imgs: 图像list
        tforms: 矩阵list
        shape: 初始的外界约束最小尺寸

    Returns:

    """
    # x = np.array([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    left_up_border, right_down_border = np.array([0, 0]), np.array([shape[1], shape[0]])

    for i, data in enumerate(imgs):
        if tforms[i] is None:
            continue
        shape = imgs[i].shape[:2]
        x = np.array([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
        y = tforms[i].inverse(x)
        a = (np.min(y[:, 0]), np.min(y[:, 1]))
        left_up_border = np.where(a < left_up_border, a, left_up_border)
        b = (np.max(y[:, 0]), np.max(y[:, 1]))
        right_down_border = np.where(b > right_down_border, b, right_down_border)
    left_up_border = np.where(left_up_border < 0, -left_up_border, 0)
    right_down_border = (right_down_border + left_up_border)
    return np.array([left_up_border[1], left_up_border[0]]), np.array([right_down_border[1], right_down_border[0]])


def addLeftUpBorder(img, border):
    """
    左上角增加黑边
    :param img: shape=(a,b)
    :param border: (a,b)
    :return:
    """
    if len(img.shape) > 2:
        tmp = np.zeros((border[0] + img.shape[0], border[1] + img.shape[1], img.shape[2]), np.uint8)
    else:
        tmp = np.zeros((border[0] + img.shape[0], border[1] + img.shape[1]), np.uint8)
    tmp[border[0]:border[0] + img.shape[0], border[1]:border[1] + img.shape[1]] = img
    return tmp


def addBorder(img, border, left_up_border):
    if len(img.shape) > 2:
        tmp = np.zeros((border[0], border[1], img.shape[2]), np.uint8)
    else:
        tmp = np.zeros(border, np.uint8)
    tmp[left_up_border[0]:left_up_border[0] + img.shape[0], left_up_border[1]:left_up_border[1] + img.shape[1]] = img
    return tmp

def calc_tform_with_leftupBorder_abandoned(tform,left_up_border):
    x = np.array([[0, 0]])
    x = tform.__call__(x)
    x[0] = x[0] + left_up_border
    y = tform.inverse(x)[0]
    form=skimage.transform.AffineTransform(translation=(-y[1],-y[0]))
    return tform.__add__(form)

def calc_tform_with_leftupBorder(tform,left_up_border):
    x = np.array([[0, 0]])
    x = tform.inverse(x)
    x[0] = x[0] + [left_up_border[1],left_up_border[0]]
    y = tform.__call__(x)[0]
    form=skimage.transform.AffineTransform(translation=(-y[0],-y[1]))
    return tform.__add__(form)


def addLeftUpBorder_cv_inverse(img, left_up_border,right_down_border, tform):
    x = np.array([[0, 0]])
    x = tform.__call__(x)
    x[0] = x[0] + left_up_border
    y = tform.inverse(x)[0]
    # print(left_up_border, y)
    # print(y)
    form=skimage.transform.AffineTransform(translation=(-y[1],-y[0]))

    return skimage.transform.warp(img,form,output_shape=right_down_border.astype(np.int32))

    # return cv2.copyMakeBorder(img, max(0, y[0]), 0, max(0, y[1]), 0, cv2.BORDER_CONSTANT)


def addBorder_cv(img, right_down_border, left_up_border,cval=0.):
    # x = left_up_border.astype(np.int32)
    # y = np.array([max(0, border[0] - img.shape[0] - left_up_border[0]),
    #               max(0, border[1] - img.shape[1] - left_up_border[1])]).astype(np.int_)
    # return cv2.copyMakeBorder(img, x[0], y[0], x[1], y[1], cv2.BORDER_CONSTANT)
    y=left_up_border
    form = skimage.transform.AffineTransform(translation=(-y[1], -y[0]))
    return (skimage.transform.warp(img, form, output_shape=right_down_border.astype(np.int32),cval=cval,preserve_range=True)).astype(np.uint8)

def simple_nms(scores, nms_radius: int):
    '''非极大抑制'''
    assert nms_radius >= 0
    scores = scores.reshape((1, scores.shape[0], scores.shape[1]))
    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)
    
    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    # c = time()
    # for _ in range(2):
    #     supp_mask = max_pool(max_mask.float()) > 0
    #     supp_scores = torch.where(supp_mask, zeros, scores)
    #     new_max_mask = supp_scores == max_pool(supp_scores)
    #     max_mask = max_mask | (new_max_mask & (~supp_mask))
    # d = time()
    # ic(d-c)
    return torch.where(max_mask, scores, zeros)[0]



def nms_map2kpts(nms_map):
    '''将热力图的形式转换成坐标点列表的形式'''
    ys = nms_map[0]
    xs = nms_map[1]
    ys = np.reshape(ys, (ys.shape[0], 1))
    xs = np.reshape(xs, (xs.shape[0], 1))
    t = torch.from_numpy(np.concatenate((xs,ys), axis=1))
    return [t]

def nms_pool(data0, data1, nums_radius):
    '''非极大抑制的池化版本, 考虑了置信度, 但是较慢'''
    kpts0 = data0['keypoints0'][0].cpu().numpy()
    kpts1 = data1['keypoints1'][0].cpu().numpy()
    # 使用最大池化的方法找局部最大值
    scores_map0 = np.zeros(data0['image0'].shape[2:])
    scores_map1 = np.zeros(data1['image1'].shape[2:])
    scores_map0[kpts0[:,1].astype(np.int64), kpts0[:,0].astype(np.int64)] = np.array(data0['scores0'][0])
    scores_map1[kpts1[:,1].astype(np.int64), kpts1[:,0].astype(np.int64)] = np.array(data1['scores1'][0])
    # 每一个像素的值都是关键点的索引, 索引映射到坐标, 不是关键点的像素值为一个很大的数(一定会超出索引范围)
    # 方便通过坐标来获取score和descriptor
    idx_map0 = np.ones(scores_map0.shape) * scores_map0.shape[0] * scores_map0.shape[1]
    idx_map0[kpts0[:,1].astype(np.int32), kpts0[:,0].astype(np.int32)] = np.arange(kpts0.shape[0])
    idx_map1 = np.ones(scores_map1.shape) * scores_map1.shape[0] * scores_map1.shape[1]
    idx_map1[kpts1[:,1].astype(np.int32), kpts1[:,0].astype(np.int32)] = np.arange(kpts1.shape[0])

    nms_map0 = np.where(simple_nms(torch.from_numpy(scores_map0), nums_radius) > 0)    
    nms_map1 = np.where(simple_nms(torch.from_numpy(scores_map1), nums_radius) > 0)

    idxs0 = idx_map0[nms_map0[0], nms_map0[1]]
    idxs1 = idx_map1[nms_map1[0], nms_map1[1]]
    data0['keypoints0'] = nms_map2kpts(nms_map0)
    data1['keypoints1'] = nms_map2kpts(nms_map1)
    data0['scores0'] = [data0['scores0'][0][idxs0]]
    data1['scores1'] = [data1['scores1'][0][idxs1]]
    data0['descriptors0'] = [data0['descriptors0'][0][:,idxs0]]
    data1['descriptors1'] = [data1['descriptors1'][0][:,idxs1]]

    return data0, data1

def filt_point(keypoints, scores, scale):
    '''删除距离过近的点, 返回保留的点的下标
    scale一般大于1, 也就是缩小尺寸,
    缩小尺寸后重合多余的点被删除.
    因此scale小于1无意义
    '''
    keypoints = keypoints / scale
    keypoints = keypoints.astype(np.int64)
    N = keypoints.shape[0]
    # 为每一个坐标计算一个唯一的整数code, 如果坐标重复, code也必然重复, 减少去重的复杂度 
    max_num = keypoints.max()+1
    kpt_codes = keypoints[:,0] * max_num + keypoints[:,1]
    kpt_codes = np.reshape(kpt_codes, (N, 1))
    scores = np.reshape(scores, (N, 1))
    # 拼接置信度列
    kpt_codes = np.concatenate((kpt_codes, scores), axis=1)
    # 对每一个点标号, 拼接后, 每一个元素应该是[code, score, id]
    idx_ = np.arange(N).reshape(N, 1)
    kpt_codes = np.concatenate((kpt_codes, idx_), axis=1)
    
    l_ = kpt_codes.tolist()
    l_.sort(key=lambda s:(s[0], s[1]), reverse=True)
    
    code_set = set()
    ids = []
    for i in l_:
        if(i[0] in code_set):
            pass
        else:
            code_set.add(i[0])
            # 记录下标
            ids.append(i[2])
    
    return ids 

def nms_scale(data0, data1, scale):
    '''使用缩放法进行非极大抑制, 较快'''
    kpts0 = data0['keypoints0'][0].cpu().numpy()
    kpts1 = data1['keypoints1'][0].cpu().numpy()
    scores0 = data0['scores0'][0].cpu().numpy()
    scores1 = data1['scores1'][0].cpu().numpy()
    # plt.imshow(data0['image0'][0,0,:,:])
    # plt.plot(data0['keypoints0'][0][:,0], data0['keypoints0'][0][:,1], '.', color='r')
    # plt.show()

    idxs0 = filt_point(kpts0, scores0, scale)
    idxs1 = filt_point(kpts1, scores1, scale)

    data0['keypoints0'] = [data0['keypoints0'][0][idxs0]]
    data1['keypoints1'] = [data1['keypoints1'][0][idxs1]]
    data0['scores0'] = [data0['scores0'][0][idxs0]]
    data1['scores1'] = [data1['scores1'][0][idxs1]]
    data0['descriptors0'] = [data0['descriptors0'][0][:,idxs0]]
    data1['descriptors1'] = [data1['descriptors1'][0][:,idxs1]]

    # plt.imshow(data0['image0'][0,0,:,:])
    # plt.plot(data0['keypoints0'][0][:,0], data0['keypoints0'][0][:,1], '.', color='r')
    # plt.show()

    return data0, data1

# def addLeftUpBorder_use_ski_affine(img1, img2,tform):
#     border = getLeftUpBorder_with_ski(tform)[0]
#     if border[0] > 0 or border[1]>0:
#         border=np.array((border[1],border[0]))
#         border = np.where(border > 0, border, 0)
#         img1 = addLeftUpBorder(img1, border)
#         img2 = addLeftUpBorder(img2, border)
#     return img1, img2
