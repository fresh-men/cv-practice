from Register import SuperglueReg
import os
def getSuperglueInstance():
    superpoint_model_path = './weights/superglue_indoor.pth'
    superglue_model_path = './weights/superpoint_v1.pth'
    device = 'cuda:0'

    superglue = SuperglueReg(
    ve_model_path=None,
    ve_model_type=None,
    superpoint_model_path=superpoint_model_path,
    superglue_model_path=superglue_model_path,
    logger=None,
    device=device,  # gpu还没测试，测试好了有的话最好用gpu，如'cuda:0'
    ve_gan_batch_size=1,  # 图像预处理的线程数，区别不大
    nms_radius=4,  # 非极大值抑制参数， 可不改
    keypoint_threshold=0.02,  # 特征点提取阈值，可不改
    max_keypoints=250,  # 最大特征点数，可不改
    sinkhorn_iterations=20,  # 迭代次数，不改
    match_threshold=0.02,  # 配对阈值，不改
    do_debug=False
    # 以上参数除了tar_size默认-1，别的都设置了如上的默认参数，调用时不改可不写
    )
    return superglue