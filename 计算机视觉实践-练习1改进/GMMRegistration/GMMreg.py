import copy
import numpy as np
import time
from GMMRegistration.kdtree import KDTree
from scipy.optimize import minimize
from GMMRegistration.keypoints2XY import enlargeX, shrinkX, standardizationXY
from GMMRegistration.formulas import *
from GMMRegistration.ShowResult import drawResPoints, drawMatchResult, warpImg, warpPerspective
from icecream import ic
class GMM:
    def __init__(self, tau=0.9, lamda=1000, eta=0.5, gamma=0.9, beta=0.1, maxiter=15) -> None:
        self.τ = tau
        self.λ = lamda
        self.η = eta
        self.γ = gamma
        self.β = beta
        self.maxiter = maxiter

    def setImgAndKeypoints(self, img1, img2, keypoints1, keypoints2, conf, inliners):
        self.img1 = img1
        self.img2 = img2
        self.keypoints1 = keypoints1
        self.keypoints2 = keypoints2
        self.conf = conf
        self.inliners = inliners

    def do_registration(self, showAdd=False):
        img1_ = self.img1.copy()
        img2_ = self.img2.copy()
        start = time.time()
        
        N = len(self.keypoints1)
        M = len(self.keypoints2)
        idx_ = np.arange(N).reshape(N, 1)
        #对 X 的每一个元素标号, 标号放在第三个元素, 方便kdtree查找邻居的编号
        X = list(np.concatenate((self.keypoints1, idx_), axis=1))
        Y = list(self.keypoints2)
        L = int((N+M)/20) + 1
        # K, L 至少为4个
        L = max(L, 4)
        K = L
        
        # 前面几个, 也就是known_corr_num个点是阈值筛选出的点,认为是可靠的
        # π_mn每一行之和为1, 总和显然是M
        π_mn = np.ones((M,N)) / N
        π_mn[self.inliners, :] = (1 - self.τ)/(N - 1)
        π_mn[self.inliners, self.inliners] = self.τ


        origY = copy.deepcopy(Y)
        origX = copy.deepcopy(X)
        # cv2.imshow("match", img1)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        # 数据标准化,0均值,1方差
        X, Y, Yxmean, Yxstd, Yymean, Yystd, Xxmean, Xxstd, Xymean, Xystd, maxYarea = standardizationXY(X, Y, self.img1.shape, self.img2.shape)
        ktimes=1000
        X = enlargeX(X,ktimes)
        # X是 3 维点集, 只取前两维, 即xy坐标
        kdTree = KDTree(X, 2)
        # 查找每个x的 K 个最近邻
        # 注意, 前面有 2 的限制, 此处的返回结果中第 3 维, 即点的索引没有改变, 直接返回, 实现了索引的确定
        neighbors = []
        for i in range(N):
            searchRes = (kdTree.knn(X[i], K+1))
            # 第一个结果是自己, 要去掉
            neighbors.append(np.array([[si[0][0]/ktimes, si[0][1]/ktimes, int(si[0][2])] for si in searchRes])[1:])
        
        X = shrinkX(X, ktimes)

        # 定义 W = N×N
        # 原文中, 若 xj 不属于 xi 的 K 个邻居, Wij = 0
        # 那么要得出最优的 W, 只考虑 xi 的 K 个邻居即可
        W = np.zeros((N,N))
        # N 个 Wi 分开求解
        for i in range(N):
            Wi = np.array(range(K))
            # 初始化 W
            nbj=0
            for nb in neighbors[i]:
                # j = int(nb[2])
                Wi[nbj] = 1/K
                nbj += 1
            fmin = minimize(fun=costW,
                            x0=Wi,args=(X, i, neighbors[i]),
                            method='SLSQP',
                            # jac=gradient,
                            constraints=({'type': 'eq', 'fun': lambda x: 1 - np.sum(x)}),
                        options={'maxiter': 100})
            # fmin = optimize.fmin_sl
            # print(fmin)
            Wi = fmin.x
            nbj=0
            for nb in neighbors[i]:
                j = int(nb[2])
                W[i][j] = Wi[nbj]
                nbj += 1
        
        # 取消最后一维索引
        # X = (x1, ··· , xN).T, (N,2)
        # Y = (y1, ... , yM).T, (M,2)
        X = np.array([x[0:2] for x in X])

        # 初始化C = 0
        # Cs由M-step进行更新, 由于采用 fast implementation, 更新时将解出 Cs 作为参数矩阵
        Cs = np.zeros((L,2))
        
        # fast implementation
        # 随机选取, 坐标不可重复
        X_=[]
        randIds = np.random.choice(range(N), size=L, replace=False)
        for n in randIds:
            X_.append(X[n])
        
        X_ = np.array(X_)
        Y = np.array(Y)
        origY = np.array(origY)
        ic(X.shape)
        ic(Y.shape)
        ic(X_.shape)

        xns=np.reshape(X,(N,2,1))
        xns = np.repeat(xns,L,axis=2)
        xndis2 = np.sum(np.square(xns-X_.T), axis=1, keepdims=True)
        kernel = np.exp(-self.β * xndis2)
        E = np.reshape(kernel, (N,L))

        """ 定义a, Set a to the volume of the output space
        参考Robust Feature Matching for Remote Sensing Image Registration via Locally Linear Transforming
        3页Problem Formulation部分,1/a denotes the outlier uniform distribution 
        with 'a' being the area of the second image (i.e., the range of yn). 
        """
        max_y = np.max(origY, axis=0)
        min_y = np.min(origY, axis=0)
        # a = (max_y[0]-min_y[0]) * (max_y[1]-min_y[1]) / ystd / ystd
        a = M
        ic('a :', a, 'K=L :', L)

        # 初始化 p_mn = π_mn
        p_mn = copy.deepcopy(π_mn)
        
        # 初始化γ
        γ=self.γ
        # Mp = getMp(p_mn)
        Mp = (1-γ)*M
        # σ^2 = (参考M-step部分的定义)
        σ2 = getσ2_vec(M, N, X, X_, Y, p_mn, Mp, L, Cs, self.β)
        CsOld = 0
        σ2old = -100
        # EM 算法
        for iteration in range(1, self.maxiter+1):
            # E-step, 更新 p_mn
            p_mn = updatePmn_vec(X, Y, M, N, π_mn, p_mn, self.τ, Cs, γ, σ2, a, kernel, self.inliners, skip=True)
            # M-step
            Cs = updateCs(W, p_mn, E, M, N, self.λ, σ2, X, Y)
            if not Cs.any():
                return np.array([]), np.array([]), np.array([]), np.array([])
            # Mp 使用旧参数(已知参数)去计算
            Mp = getMp(p_mn)
            if(Mp == M):
                break
            # 更新 σ^2, γ   
            γ = 1 - Mp/M
            σ2 = getσ2_vec(M, N, X, X_, Y, p_mn, Mp, L, Cs, self.β)
            ic(iteration)
            ic(Mp)
            # showRes(img1, img2, csnew, X_, Xxmean, Xxstd, Xymean, Xystd, maxY, minY)
            drawResPoints(self.img1, self.img2, X, origY, Y, Cs, X_, Yxmean, Yxstd, Yymean, Yystd, iter=iteration, β=self.β)
            if(σ2old - σ2 < 0.000001 and σ2old - σ2 > -0.000001):
                break
            σ2old = σ2
            CsOld = Cs
        # EM循环结束
        drawResPoints(self.img1, self.img2, X, origY, Y, Cs, X_, Yxmean, Yxstd, Yymean, Yystd, iter=iteration, β=self.β, showFinal=showAdd)
        end = time.time()
        GMM_time = end - start
        ic(GMM_time)
        print('gmm time', GMM_time)
        from matplotlib import pyplot as plt
        return warpImg(img1_, img2_, Cs, X_, Yxmean, Yxstd, Yymean, Yystd, Xxmean, Xxstd, Xymean, Xystd, β=self.β, showAdd=showAdd)