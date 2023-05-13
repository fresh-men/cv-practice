import cv2
from matplotlib import pyplot as plt
import numpy as np
from GMMRegistration.formulas import T, getNewPositions
def warpImg(img1, img2, Cs, X_, Yxmean, Yxstd, Yymean, Yystd, Xxmean, Xxstd, Xymean, Xystd, β=0.1, showAdd=False):
    img1Shape = img1.shape
    img2Shape = img2.shape
    
    # 图像变形, 每一个像素对应一个新位置
    newy, newx, originaly, originalx = getNewPositions(img1Shape, Yxmean, Yxstd, Yymean, Yystd, Xxmean, Xxstd, Xymean, Xystd, len(X_), Cs.T, X_, β)
    max_newy = newy.max()
    max_newx = newx.max()
    min_newy = newy.min()
    min_newx = newx.min()
    # max_origy = originaly.max()
    # max_origx = originalx.max()
    max_origy = img1Shape[0]
    max_origx = img1Shape[1]
    # maxy = max(max_newy, originaly.max())
    # maxx = max(max_newx, originalx.max())
    top, bottom, right, left = 0, 0, 0, 0
    if max_newy > max_origy:
        bottom = max_newy - max_origy

    if min_newy < 0:
        top = 0 - min_newy
    
    if max_newx > max_origx:
        right = max_newx - max_origx

    if min_newx < 0:
        left = 0 - min_newx

    # 两张图片放在同一个框内
    resizedImg2 = cv2.copyMakeBorder(img2, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
    # dst = np.zeros((int(maxy+1), int(maxx+1), img2Shape[2])).astype('uint8')
    dst = np.zeros(resizedImg2.shape).astype('uint8')
    # 负数坐标会导致图片出现在两端, 转移到正区间
    if min_newy < 0:
        newy -= min_newy
        
    if min_newx < 0:
        newx -= min_newx
    # newy = np.minimum(np.maximum(newy, 0), resizedImg2.shape[0]-1)
    # newx = np.minimum(np.maximum(newx, 0), resizedImg2.shape[1]-1)
    newy = np.clip(newy, 0, resizedImg2.shape[0]-1)
    newx = np.clip(newx, 0, resizedImg2.shape[1]-1)
    dst[newy, newx] = img1[originaly, originalx]
    # 去除噪音线
    dst = switch_medianBlur(dst)
    if showAdd:
        # plt.imshow(dst)
        # plt.show()
        # plt.imshow(resizedImg2)
        # plt.show()
        added = cv2.addWeighted(dst, 0.5, resizedImg2, 0.5, 0.0)
        plt.imshow(added)
        plt.show()

    return dst, resizedImg2

def switch_medianBlur(img):
    '''
    开关中值滤波
    '''
    zero_idx = np.where(img == 0)
    median = cv2.medianBlur(img, 5)
    img[zero_idx] = median[zero_idx]
    return img

def cut_background(img):
    height, width = img.shape[0], img.shape[1]
    tmp_img = img[:, int(width / 2 - height / 2): int(width / 2 + height / 2)]
    return tmp_img

def getMask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    
    # mask_label_num, mask_label_img, mask_img_stats, _ = cv2.connectedComponentsWithStats(mask)
    # mask_label_mask_index = mask_img_stats[1:,4].argmax() + 1
    # mask = np.array((mask_label_img == mask_label_mask_index) * 255).astype(np.uint8)
    # mask = cv2.threshold(mask, 125, 255, cv2.THRESH_BINARY)[1]
    # mask = cut_background(mask)
    return mask

def drawResPoints(img1, img2, X, origY, Y, Cs, X_, Yxmean, Yxstd, Yymean, Yystd, iter = 0, β=0.1, showFinal=False):
    img1Shape = img1.shape
    # img2Shape = img2.shape
    tmp = np.zeros((600,610,3))
    if(showFinal):
        k=70
        b=150
        for ym in Y:
            ymx = int(ym[0]*k+b)
            ymy = int(ym[1]*k+b)
            if not(ymx < 0 or ymy < 0 or ymx >= tmp.shape[0] or ymy >= tmp.shape[1]):
                cv2.circle(tmp, (ymx,ymy), 1, (0,0,255), 2)
        for xn in X:
            rex = xn[0]
            rey = xn[1]
            # cv2.circle(tmp, (int(rex*k+b),int(rey*k+b)), 1, (255,0,0), 2)
            newPosition = T([rex,rey], X_, len(X_), Cs.T, β=β, output=True)
            newx = int(newPosition[0][0]*k+b)
            newy = int(newPosition[1][0]*k+b)

            if not(newx < 0 or newy < 0 or newx >= tmp.shape[0] or newy >= tmp.shape[1]):
                cv2.circle(tmp, (newx,newy), 1, (0,255,0), 2)
        # cv2.imshow("testpoint dst", tmp)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        nonz = np.nonzero(tmp)
        tmp = tmp[np.min(nonz[0]):np.max(nonz[0]), np.min(nonz[1]):np.max(nonz[1])]
        plt.imshow(tmp)
        plt.show()

    for ym in origY:
        ymx = int(ym[0])
        ymy = int(ym[1])
        if not(ymx < 0 or ymy < 0 or ymx >= img1.shape[0] or ymy >= img1.shape[1]):
            cv2.circle(img1, (ymx,ymy), 1, (0,0,255), 2)
            cv2.circle(img2, (ymx,ymy), 1, (0,0,255), 2)
    for xn in X:
        rex = xn[0]
        rey = xn[1]
        # rex = (x-Xxmean)/Xxstd
        # rey = (y-Xymean)/Xystd
        newPosition = T([rex,rey], X_, len(X_), Cs.T, β=β)
        newx = int(newPosition[0][0]*Yxstd+Yxmean)
        newy = int(newPosition[1][0]*Yystd+Yymean)
        co = 20+30*iter % 255
        if not(newx < 0 or newy < 0 or newx >= img1.shape[0] or newy >= img1.shape[1]):
            cv2.circle(img1, (newx,newy), 1, (co,co,co), 2)

def showimg(img, name='image'):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def drawMatchResult(img1, img2, p_mn, keypts1, keypts2, η):
    h1, w1, c1 = img1.shape
    h2, w2, c1 = img2.shape
    if(h1 < h2):
        '''底部补齐'''
        disH = h2 - h1
        img1 = cv2.copyMakeBorder(img1, 0,disH,0,0, 0, (0,0,0))
    elif(h1 > h2):
        disH = h1 - h2
        img2 = cv2.copyMakeBorder(img2, 0,disH,0,0, 0, (0,0,0))
    concat = np.concatenate((img1, img2), axis=1)
    
    """画BGR颜色的点
    for m in range(p_mn.shape[0]):
        for n in range(p_mn.shape[1]):
            if(p_mn[m][n] > η):
                cnt = cnt + 1
                print(m,n,':',p_mn[m][n])
                x = keypts1[n]
                y = keypts2[m]
                cv2.circle(concat, (int(x[0]),int(x[1])), 1, (255,0,0), 2)
                cv2.circle(concat, (int(y[0])+w1,int(y[1])), 1, (0,0,255), 2)
                cv2.line(concat, (int(x[0]),int(x[1])), (int(y[0])+w1,int(y[1])), (0,255,0), 1)
    """
    '''保留每一列最大值, 防止一个点有多个对应'''
    for i in range(p_mn.shape[1]):
        slice = p_mn[:,i]
        max = slice.max()
        p_mn[:,i] = np.where(slice == max, max, 0)

    ys, xs = np.where(p_mn > η)
    # print('xs {}, ys{}'.format(len(xs), len(ys)))
    resX = []
    resY = []
    for i in range(len(xs)):
        x = keypts1[xs[i]]
        y = keypts2[ys[i]]
        resX.append(x)
        resY.append(y)
        cv2.circle(concat, (int(x[0]), int(x[1])), 1, (255,0,0), 2)
        cv2.circle(concat, (int(y[0])+w1,int(y[1])), 1, (0,0,255), 2)
        cv2.line(concat, (int(x[0]),int(x[1])), (int(y[0])+w1,int(y[1])), (0,255,0), 1)
    print('保存匹配数:', len(xs))
    showimg(concat)
    return np.float32(resX), np.float32(resY)

def warpPerspective(img1, img2, resX, resY):
    resX = resX[:, :2]
    M, mask = cv2.findHomography(resX, resY, cv2.RANSAC, 5.0)
    dst = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
    # showimg(dst)
    dstAdd = cv2.addWeighted(img2, 0.5, dst, 0.5, 0.0)
    showimg(dstAdd, 'Final result')
