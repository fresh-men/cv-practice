import numpy as np
import math
from icecream import ic
# diagonal Gaussian kernel Γ
# 返回IR2 × IR2 → IR2×2
# xi, xj表示两点, 即keypoints的pt属性, 使用列向量
def getΓ(xi, xj, β):
    I = np.identity(2)
    dis2 = (xi[0][0]-xj[0][0])**2 + (xi[1][0]-xj[1][0])**2
    k = np.exp(0-β*dis2)
    return k*I

# T的建模, 即需要求解的空间变换函数
# T(x) = x + f(x), x代表原来的点位
# C = (c1,c2,...cl), x,c是二维列向量, 代表坐标和参数
# 返回一个变换后的坐标 2×1
def T(x, X_, L, Cs, β, output=False):
    sumL = np.zeros((2,1))
    x = np.reshape(x, (2,1))
    rexy = x

    for l in range(L):
        x_l = np.reshape(X_[l], (2,1))
        Γ = getΓ(rexy, x_l, β)
        cl = np.reshape(Cs[:,l], (2,1))
        # sumL = sumL + Γ * cl
        sumL = sumL + np.matmul(Γ, cl)
    # if(output):
        # print(x, sumL)
    return x + sumL
# 以矩阵形式优化上述变换, 获得所有原始坐标的新坐标
# X_:(L,2), Cs:(2,L)
def getNewPositions(srcShape, Yxmean, Yxstd, Yymean, Yystd, Xxmean, Xxstd, Xymean, Xystd, L, Cs, X_, β):
    # 需要确保通道不为空
    srcH, srcW, srcC = srcShape

    originalPos = np.zeros((srcH, srcW, L, 2))
    # 看做一个以二维坐标为元素的矩阵, 并让元素的值为其下标
    # 此处要注意, 坐标的变化很容易混淆, 要清楚初始坐标是什么形式, 对其分别进行怎么样的操作
    for tmpy in range(srcH):
        originalPos[tmpy,:,:,1] = (tmpy-Xymean)/Xystd
    for tmpx in range(srcW):
        originalPos[:,tmpx,:,0] = (tmpx-Xxmean)/Xxstd

    # 计算diagonal Gaussian kernel Γ(xi, xj)中的值, 为方便编程, 把xy轴分开计算, 最后合并
    dis2 = np.sum(np.square(originalPos-X_), axis=3, keepdims=True)
    kernel = np.exp(-β * dis2)

    fy = np.matmul(Cs[None,1], kernel)
    fy = np.reshape(fy, fy.shape[:2])

    fx = np.matmul(Cs[None,0], kernel)
    fx = np.reshape(fx, fx.shape[:2])

    originaly = originalPos[:,:,0,1]
    originalx = originalPos[:,:,0,0]

    # print('(',originalx[22,120],originaly[22,120],')',fx[22,120],fy[22,120])

    newy = (originaly + fy) * Yystd + Yymean
    newx = (originalx + fx) * Yxstd + Yxmean
    
    # 不考虑溢出了, 放在别的函数去处理
    newy = newy.astype(np.int_).flatten()
    newx = newx.astype(np.int_).flatten()

    # 生成数组形式的原xy坐标
    unit_ = np.arange(srcH)
    originaly = np.repeat(unit_, srcW)
    unit_ = np.arange(srcW)
    originalx = np.tile(unit_, srcH)
    return newy, newx, originaly, originalx

# 若 nid != None, 说明要取其中的第nid个分模型的输出, 否则输出全部的和
# X, Y就是点xn, ym的集合
def P(ymid, N, γ, a, π_mn, σ2, Y, X, X_, L, Cs, β, nid=None):
    sumN = 0
    if(nid==None):
        for n in range(N):
            # 转换后的坐标
            x_t = T(X[n], X_, L, Cs.T, β)
            # 距离的平方
            distance2 = (Y[ymid][0]-x_t[0][0])**2 + (Y[ymid][1]-x_t[1][0])**2
            sumN = sumN + (π_mn[ymid][n] / (2*np.pi*σ2))*np.exp(0 - distance2/2/σ2)
        
    else:
        x_t = T(X[nid], X_, L, Cs.T, β)
        distance2 = (Y[ymid][0]-x_t[0])**2 + (Y[ymid][1]-x_t[1])**2
        sumN = sumN + (π_mn[ymid][nid] / (2*np.pi*σ2))*np.exp(0 - distance2/2/σ2)
        
    return sumN*(1-γ) + γ/a

# 计算 Mp, Mp <= M
def getMp(p_mn):
    return np.sum(p_mn)

def getσ2(M, N, X, X_, Y, p_mn, Mp, L, Cs, β):
    sumMN = 0
    for m in range(M):
        for n in range(N):
            xnt = T(X[n].T, X_, L, Cs.T, β)
            # 预期为 2*1 的列向量
            # print("xnt:",xnt.shape)
            xnt = xnt.T
            dis2 = (Y[m][0] - xnt[0][0])**2 + (Y[m][1] - xnt[0][1])**2
            # if(p_mn[m][n] > 0.7):
                # print('dis btw ym and T(xn)',m,n,dis2)
                # dis_ = (X[n][0] - xnt[0][0])**2 + (X[n][1] - xnt[0][1])**2
                # print('dis btw xn and T(xn', dis_)
            sumMN = sumMN + p_mn[m][n] * dis2
    
    return sumMN / 2 / Mp
# 用向量化优化上述算法
def getσ2_vec(M, N, X, X_, Y, p_mn, Mp, L, Cs, β):
    # 计算所有T(Xn)
    xns=np.reshape(X,(N,2,1))
    xns = np.repeat(xns,L,axis=2)
    xndis2 = np.sum(np.square(xns-X_.T), axis=1, keepdims=True)
    kernel = np.exp(-β * xndis2)

    fy = np.matmul(kernel, Cs[:,1,None])
    fy = np.reshape(fy,fy.shape[:1])
    fx = np.matmul(kernel, Cs[:,0,None])
    fx = np.reshape(fx,fx.shape[:1])

    newy = X[:,1] + fy
    newx = X[:,0] + fx
    newxy = np.vstack((newx,newy))

    # 计算每一个ym和所有xn的距离平方, 结果显然是M*N的矩阵
    yms = np.reshape(Y,(M,2,1))
    yms = np.repeat(yms,N,axis=2)
    xydis2 = np.sum(np.square(yms-newxy).transpose((0,2,1)), axis=2)

    # 计算乘积的和(点积, 对应位置相乘相加即可)
    mult = np.dot(p_mn.flatten(), xydis2.flatten())
    return mult / 2 / Mp

def gaussian_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)

# 初始 Wi = 0, 1 × N
# X, neighbors 都是三维, 第三维表示索引
def costW(Wi, X, xid, neighbors):
    xi = X[xid]
    # print(Wi)
    sumj = np.zeros((1,2))
    # 只考虑 K 个邻居
    nbj=0
    for n in neighbors:
        # j = int(n[2])
        xj = np.array(n[0:2])
        sumj = sumj + xj * Wi[nbj]
        nbj += 1
    # print(xi, '*')
    # print(sumj)
    res = (xi[0] - sumj[0][0])**2 + (xi[1] - sumj[0][1])**2
    # print(res)
    return res

def gradient(Wi, X, xid, neighbors):
    grad = np.zeros(Wi.shape)
    # 只考虑 K 个邻居
    # xy坐标
    sumWx = 0
    sumWy = 0
    for n in neighbors:
        j = int(n[2])
        xj = np.array(n[0:2])
        sumWx = sumWx + Wi[j]*xj[0]
        sumWy = sumWy + Wi[j]*xj[1]
    for n in neighbors:
        j = int(n[2])
        xj = np.array(n[0:2])
        # grad[j] = 2 * (xj[0]-X[xid][0] + xj[1]-X[xid][1])
        grad[j] = 2 * (xj[0] * (0 - X[xid][0] + sumWx) - xj[1] * (X[xid][1] + sumWy))
    return grad

# 参照公式 3, 一个对数似然函数
# σ2, γ都是新参数, Mp使用旧参数计算
def Qtheta(Mp, σ2, γ, M, N, X, Y, X_, L, C, β, p_mn):
    former = Mp * (np.log(σ2) - np.log(1-γ)) - np.log(γ) * (M - Mp)
    print('former:',Mp,'*','(',np.log(σ2), '-', np.log(1-γ),')', '-', np.log(γ), '*', (M - Mp),'=',former)
    sumMN = 0
    for m in range(M):
        for n in range(N):
            # P(ymid, N, γ, a, π_mn, σ2, Y, X, X_, L, C, β, nid=None)
            # pYmZn = P(m, N, γ, a, π_mn, σ2, Y, X, X_, L, C, β, nid=n)
            # pYm = P(m, N, γ, a, π_mn, σ2, Y, X, X_, L, C, β)
            # Pzmn = π_mn[m][n]*pYmZn/pYm
            # 转换后的坐标
            x_t = T(X[n], X_, L, C.T, β)
            # 距离的平方
            distance2 = (Y[m][0]-x_t[0][0])**2 + (Y[m][1]-x_t[1][0])**2
            sumMN = sumMN + p_mn[m][n] * distance2
    latter = sumMN / 2 / σ2
    return former + latter, latter

def updatePmn(X, Y, M, N, π_mn, p_mn, τ, X_, L, Cs, β, γ, σ2, a, skip=True):
    # 此处对 M 进行了遍历, 而没有考虑已经确定的对应点. 此处参考 E-step 下面的解释
    for m in range(M):
        conti = False
        if(skip):
            # for pn in p_mn[m]:
            #     if(pn >= τ):
            #         conti = True
            #         break
            for n in range(N):
                if(π_mn[m][n] >= τ):
                    conti = True
                    break
        if(conti):
            continue
        # 分母中的累加部分
        sumN = 0
        π_exps = []
        for k in range(N):
            x_t = T(X[k], X_, L, Cs.T, β)
            # print(x_t)
            distance2 = (Y[m][0]-x_t[0][0])**2 + (Y[m][1]-x_t[1][0])**2
            π_exps.append(π_mn[m][k] * np.exp(0 - distance2/2/σ2))
        
        sumN = np.sum(π_exps)    
        # print(sumN, (2*γ*np.pi*σ2)/(1-γ)/a)
        denominator = sumN + (2*γ*np.pi*σ2)/(1-γ)/a
        for n in range(N):
            # if(skip and p_mn[m][n] >= τ):
            #     continue
            # 此处的返回是列向量, 注意操作
            x_t = T(X[n], X_, L, Cs.T, β)
            numerator = π_exps[n]
            # print(numerator)
            p_mn[m][n] = numerator / denominator
    
    return p_mn

# 用向量化优化上述算法
def updatePmn_vec(X, Y, M, N, π_mn, p_mn, τ, Cs, γ, σ2, a, kernel, inliners, skip=True):
    # 计算所有T(Xn), 即新坐标newxy
    fy = np.matmul(kernel, Cs[:,1,None])
    fy = np.reshape(fy,fy.shape[:1])
    newy = X[:,1] + fy

    fx = np.matmul(kernel, Cs[:,0,None])
    fx = np.reshape(fx,fx.shape[:1])
    newx = X[:,0] + fx

    newxy = np.vstack((newx,newy))

    # 计算每一个ym和所有xn的距离平方, 结果显然是M*N的矩阵
    yms = np.reshape(Y,(M,2,1))
    yms = np.repeat(yms,N,axis=2)
    xydis2 = np.sum(np.square(yms-newxy).transpose((0,2,1)), axis=2)
    
    # 得到自然常数项
    exps = np.exp(xydis2 / (-2*σ2))
    
    # 分子部分
    numerator = π_mn * exps

    # 分母部分, 经过观察, 分母仅与m有关, 因此算出shape为(M,1)的分母后, 按行相除即可
    denominator = np.sum(numerator, axis=1, keepdims=True) + (2*γ*np.pi*σ2)/(1-γ)/a

    p_mn_new = numerator / denominator

    # 体现半监督的位置, 已经确定了对应关系的ym和xn的对应的p_mn需要保持原样
    if(skip):
        p_mn[inliners, inliners] = τ

    return p_mn_new

def updateCs(W, p_mn, E, M, N, λ, σ2, X, Y):
    # 更新 Cs
    # PT*1 : N * 1 对角变成 N * N
    # reshape只是为了方便变成对角,取[0]变成一个一维列表, 包含对角上的数字
    # 根据参考文献中P(与本文不同)的形式, dPT1可能与该论文中的P(N*N的对角阵)作用相同
    one = np.ones((M, 1))
    ET = E.T
    PT = p_mn.T
    dPT1 = np.diag(np.reshape(np.matmul(PT, one), (1,N))[0])
    # 参考Robust Feature Matching for Remote Sensing Image Registration via Locally Linear Transforming, 页5
    _I = np.identity(W.shape[0])
    _sub = _I - W
    Q = np.matmul(_sub.T, np.matmul(dPT1, _sub))
    bracket = dPT1 + 2*λ*σ2*Q
    A = np.matmul(np.matmul(ET, bracket), E)
    B = np.matmul(np.matmul(ET, PT), Y) - np.matmul(np.matmul(ET, bracket), X)
    
    # 求解 A*Cs=B
    # 此处可能发生矩阵不可逆的异常, 导致求解失败
    # 由行列式的性质, |A| = 0 则不可逆, 可能是由于A中存在两行完全一样的数, 一般说明随机选取的 X_ 出现了重复
    # 也有可能是算法的超参数原因, 导致A中的参数非常小, 接近于0
    try:
        Cs = np.linalg.solve(A, B)
    except Exception as e:
        print("linalg error: ", e)
        return np.array([])
        
    return Cs

def negativeLogTheta(M, N, γ, a, π_mn, σ2, Y, X, X_, L, Cs, β):
    sumM = 0
    for m in range(M):
        pym = P(m, N, γ, a, π_mn, σ2, Y, X, X_, L, Cs, β, nid=None)
        # if(pym>=1 or pym<0):
        #     print(m, pym, '------**------')
        sumM = sumM - np.exp(pym)
    
    return sumM

