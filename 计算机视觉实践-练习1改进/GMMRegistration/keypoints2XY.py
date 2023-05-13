import numpy as np
import cv2
def keypoints2XY_test(img1, keypoints1, img2, keypoints2, known_corr, M, N, divisionThreshold):
    X = []
    Y = []
    xset = set()
    yset = set()
    tmpId = 0
    known_corr_num = 0
    th, thmat1 = cv2.threshold(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), divisionThreshold, 255, type=cv2.THRESH_BINARY)
    th, thmat2 = cv2.threshold(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), divisionThreshold, 255, type=cv2.THRESH_BINARY)
    ker_ = np.ones((5,5), np.uint8)
    mask1 = cv2.erode(thmat1, ker_, iterations=6)
    mask2 = cv2.erode(thmat2, ker_, iterations=6)
    
    # 注意, Q矩阵并没有在本论文中定义, 而在引用中, M = N, 因此此处使二者相等
    # 这意味着可能发生关键点不足 N 个的异常, 不过一般情况下SIFT检测会产生多于100个点
    # if(len(keypoints1) < N+50 or len(keypoints2) < M+50):
    #     print(len(keypoints1), len(keypoints2))
    #     print("可用关键点不足")
    #     return "可用关键点不足"
    
    # 这一步 X 是 (x, y, idx), N × 3
    # 此处注意, 删去了一部分坐标重复的点
    for _mn in known_corr:
        _m = _mn[0]
        _n = _mn[1]
        kpxpt = keypoints1[_n].pt
        kpypt = keypoints2[_m].pt
        # print(mask1[int(kpxpt[1]), int(kpxpt[0])])
        if(mask1[int(kpxpt[1]), int(kpxpt[0])] <= divisionThreshold or mask2[int(kpypt[1]), int(kpypt[0])] <= divisionThreshold):
            continue

        if (xset.issuperset([kpxpt]) or yset.issuperset([kpypt])):
            continue
        else:
            xset.add(kpxpt)
            yset.add(kpypt)
            tmplist = list(kpxpt)
            tmplist.append(tmpId)
            X.append(tmplist)
            Y.append(list(kpypt))
            tmpId = tmpId + 1
            known_corr_num = known_corr_num + 1
    
    tmpidY = tmpId
    # 尝试使用可靠的点的附近的未知匹配的点
    xstart = known_corr[0][1]
    for i in range(xstart, len(keypoints1)):
        kpx = keypoints1[i]
        kpxpt = kpx.pt
        # cv2.circle(mask1, (int(kpxpt[0]),int(kpxpt[1])), 1, (255,255,255), 2)
        if(mask1[int(kpxpt[1])][int(kpxpt[0])] <= divisionThreshold):
            # cv2.circle(mask1, (int(kpxpt[1]),int(kpxpt[0])), 1, (255,255,255), 2)
            continue
        if(N <= tmpId):
            break
        xy = kpx.pt
        if not(xset.issuperset([xy])):
            xset.add(xy)
            tmplist = list(xy)
            tmplist.append(tmpId)
            X.append(tmplist)
#         print(X[tmpId])
            tmpId = tmpId + 1
    # 如果没有选够N个, 从头选
    # cv2.imshow("...", mask1)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    if(tmpId < N):
        for kpx in keypoints1:
            if(N <= tmpId):
                break
            xy = kpx.pt
            if(mask1[int(xy[1])][int(xy[0])] <= divisionThreshold):
            # cv2.circle(mask1, (int(kpxpt[1]),int(kpxpt[0])), 1, (255,255,255), 2)
                continue
            if not(xset.issuperset([xy])):
                xset.add(xy)
                tmplist = list(xy)
                tmplist.append(tmpId)
                X.append(tmplist)
    #         print(X[tmpId])
                tmpId = tmpId + 1
    tmpId = tmpidY
    ystart = known_corr[0][0]
    for j in range(ystart, len(keypoints2)):
        kpy = keypoints2[j]
        kpypt = kpy.pt
        if(mask2[int(kpypt[1])][int(kpypt[0])] <= divisionThreshold):
            continue
        if(M <= tmpId):
            break
        xy = kpy.pt
        if not(yset.issuperset([xy])):
            yset.add(xy)
            Y.append(list(xy))
            tmpId = tmpId + 1
    # 同理, 从头选
    if(tmpId < M):
        for kpy in keypoints2:
            if(M <= tmpId):
                break
            xy = kpy.pt
            if(mask2[int(xy[1])][int(xy[0])] <= divisionThreshold):
                continue
            if not(yset.issuperset([xy])):
                yset.add(xy)
                Y.append(list(xy))
                tmpId = tmpId + 1
    
    return X, Y, known_corr_num

def keypoints2XY(img1, keypoints1, img2, keypoints2, known_corr, selected_x_set, selected_y_set, divisionThreshold):
    # divisionThreshold用于分割眼底图像外围的黑色遮罩部分, 因此一般设置的值较低
    X = []
    Y = []
    xset = set()
    yset = set()
    # 用于标记每个x点的索引
    tmpId = 0
    known_corr_num = 0
    th, thmat1 = cv2.threshold(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), divisionThreshold, 255, type=cv2.THRESH_BINARY)
    th, thmat2 = cv2.threshold(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), divisionThreshold, 255, type=cv2.THRESH_BINARY)
    ker_ = np.ones((5,5), np.uint8)
    mask1 = cv2.erode(thmat1, ker_, iterations=6)
    mask2 = cv2.erode(thmat2, ker_, iterations=6)
    # 这一步 X 是 (x, y, idx), N × 3
    # 此处注意, 删去了一部分坐标重复的点
    # 确保已经认为可靠的点仅排列在前known_corr_num个
    for _mn in known_corr:
        _m = _mn[0]
        _n = _mn[1]
        kpxpt = keypoints1[_n].pt
        kpypt = keypoints2[_m].pt

        if(mask1[int(kpxpt[1]), int(kpxpt[0])] <= divisionThreshold or mask2[int(kpypt[1]), int(kpypt[0])] <= divisionThreshold):
            continue

        if (xset.issuperset([kpxpt]) or yset.issuperset([kpypt])):
            continue
        else:
            xset.add(kpxpt)
            yset.add(kpypt)
            tmplist = list(kpxpt)
            tmplist.append(tmpId)
            X.append(tmplist)
            Y.append(list(kpypt))
            tmpId = tmpId + 1
            known_corr_num = known_corr_num + 1
    for m in selected_y_set:
        kpypt = keypoints2[m].pt
        if(mask2[int(kpypt[1]), int(kpypt[0])] <= divisionThreshold):
            continue

        if(yset.issuperset([kpypt])):
            continue

        yset.add(kpypt)
        Y.append(list(kpypt))

    for n in selected_x_set:
        kpxpt = keypoints1[n].pt
        if(mask1[int(kpxpt[1]), int(kpxpt[0])] <= divisionThreshold):
            continue

        if(xset.issuperset([kpxpt])):
            continue

        xset.add(kpxpt)
        tmplist = list(kpxpt)
        tmplist.append(tmpId)
        X.append(tmplist)
        tmpId = tmpId + 1

    return X, Y, known_corr_num
            
def standardizationXY(X,Y,shape1, shape2):
    Xxs = [p[0] for p in X]
    Xys = [p[1] for p in X]
    Yxs = [p[0] for p in Y]
    Yys = [p[1] for p in Y]
    N = len(X)
    M = len(Y)
    Xxmean = np.mean(Xxs)
    Xymean = np.mean(Xys)
    Yxmean = np.mean(Yxs)
    Yymean = np.mean(Yys)
    Xxstd = np.std(Xxs)
    Xystd = np.std(Xys)
    Yxstd = np.std(Yxs)
    Yystd = np.std(Yys)
    maxYx = 0
    minYx = 0
    maxYy = 0
    minYy = 0
    for n in range(N):
        X[n][0] = (X[n][0] - Xxmean) / Xxstd
        X[n][1] = (X[n][1] - Xymean) / Xystd
        # X[n][0] = X[n][0] / shape1[0]
        # X[n][1] = X[n][1] / shape1[1]

    for m in range(M):
        Y[m][0] = (Y[m][0] - Yxmean) / Yxstd
        Y[m][1] = (Y[m][1] - Yymean) / Yystd
        # Y[m][0] = Y[m][0] / shape2[0]
        # Y[m][1] = Y[m][1] / shape2[1]
        if(Y[m][0] > maxYx):
            maxYx = Y[m][0]
        elif(Y[m][0] < minYx):
            minYx = Y[m][0]
        
        if(Y[m][1] > maxYy):
            maxYy = Y[m][1]
        elif(Y[m][1] < minYy):
            minYy = Y[m][1]
    
    maxYarea = (maxYx - minYx) * (maxYy - minYy)
    # print(maxYx, maxYy, minYx, minYy, maxYarea)
    # aaa = 1/0
    return X, Y, Yxmean, Yxstd, Yymean, Yystd, Xxmean, Xxstd, Xymean, Xystd, maxYarea

def enlargeX(X, k):
    # X三维
    for x in X:
        x[0] = x[0] * k
        x[1] = x[1] * k
    
    return X

def shrinkX(X, k):
    # X三维
    for x in X:
        x[0] = x[0] / k
        x[1] = x[1] / k
    
    return X

def getXYbyTestPoint(keypoints1, keypoints2, known_corr, τ):
    X = np.load('pointX.npy').tolist()
    Y = np.load('pointY.npy').tolist()
    xset = set()
    yset = set()
    tmpId = len(X)
    known_corr_num = 0
    # 这一步 X 是 (x, y, idx), N × 3
    # 此处注意, 删去了一部分坐标重复的点
    for _mn in known_corr:
        _m = _mn[0]
        _n = _mn[1]
        kpxpt = keypoints1[_n].pt
        kpypt = keypoints2[_m].pt
        if (xset.issuperset([kpxpt]) or yset.issuperset([kpypt])):
            continue
        else:
            xset.add(kpxpt)
            yset.add(kpypt)
            tmplist = list(kpxpt)
            tmplist.append(tmpId)
            X.append(tmplist)
            Y.append(list(kpypt))
            tmpId = tmpId + 1
            known_corr_num = known_corr_num + 1
    N = len(X)
    M = len(Y)
    π_mn = np.ones((M,N)) / N
    for k in range(known_corr_num):
        π_mn[M-k-1][N-k-1] = τ
        for n_N in range(N):
            if(n_N == N-k-1):
                continue
            π_mn[M-k-1][n_N] = (1 - τ)/(N - 1)
    # pimn等于M
    print('sum pi mn', np.sum(π_mn))
    # aaaa = 1/0
    return X, Y, π_mn

def getXYbyHand(τ):
    X=[[297,89,0],[327,104,1],[361,121,2],[300,39,3],[350,70,4],[388,108,5],[422,157,6],[360,34,7],[410,100,8],[428,205,9],[428,249,10],[406,300,11],[404,337,12],[374,297,13],[330,348,14],[313,310,15],[250,340,16]]
    # Y=[[298,115,0],[329,127,1],[356,141,2],[425,180,3],[389,131,4],[352,96,5],[301,64,6],[360,58,7],[411,124,8],[433,216,9],[460,196,10],[429,260,11],[430,282,12],[412,323,13],[409,357,14],[382,318,15],[335,371,16],[321,333,17],[260,364,18]]
    Y=[[298,115],[329,127],[356,141],[425,180],[389,131],[352,96],[301,64],[360,58],[411,124],[433,216],[429,260],[412,323],[409,357],[382,318],[335,371],[321,333],[260,364]]
    N = len(X)
    M = len(Y)
    π_mn = np.ones((M,N)) / N
    π_mn[1][1] = τ
    for n_N in range(N):
        if(n_N == 1):
            continue
        π_mn[1][n_N] = (1 - τ)/(N - 1)
    
    π_mn[2][2] = τ
    for n_N in range(N):
        if(n_N == 2):
            continue
        π_mn[2][n_N] = (1 - τ)/(N - 1)
    
    π_mn[16][16] = τ
    for n_N in range(N):
        if(n_N == 16):
            continue
        π_mn[16][n_N] = (1 - τ)/(N - 1)
    
    print(np.sum(π_mn))
    return X, Y, π_mn