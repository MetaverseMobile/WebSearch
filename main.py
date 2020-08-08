#! /usr/bin/env python3
# -*- coding:utf-8 -*-
# __author__ = "HX"
# __date__ = "2019-12-18"
# __modified__ = "2019-12-28"
import cv2 as cv
import numpy as np
from scipy.stats import multivariate_normal


def slide_window(image, size, cut_size, step):
    """
        从单个图像中提取特征向量
        :param image: RGB 图片像素矩阵
        :param size: 元组，像素块长和宽
        :param step: 像素块滑动步长
        :param get_len_x: 进行压缩时提取的左上角像素列数
        :param get_len_y: 进行压缩时提取的左上角像素行数
        :return: 该图像的特征向量矩阵
    """
    # 获取图像以及遍历所需参数
    y_limit, x_limit, _ = image.shape
    x_length, y_length = size

    # 转换色彩空间为 yCrCb
    img = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)

    # 要进行 DCT，必须首先要将矩阵转换为 float
    mat = np.float32(img)

    # N 表示像素块的个数
    get_len_y = cut_size[1]
    get_len_x = cut_size[0]
    N = int(((y_limit-y_length)/step)*((x_limit-x_length)/step))
    feature_vector = np.zeros([N, get_len_y*get_len_x*3], dtype=np.float64)

    # 利用滑动窗口，取图像块
    k = 0
    for i in range(0, y_limit-y_length, step):
        for j in range(0, x_limit-x_length, step):
            # 依次对块进行离散余弦变换
            x_y = cv.dct(mat[i:i+y_length, j:j+x_length, 0])
            x_b = cv.dct(mat[i:i+y_length, j:j+x_length, 1])
            x_r = cv.dct(mat[i:i+y_length, j:j+x_length, 2])

            # 由于余弦变换后，能量主要集中在矩阵左上角
            # 经过观察，选取每个 8x8 像素块左上角的 4x4
            # 个像素点作为参考
            x_y_compressed = x_y[0:get_len_y, 0:get_len_x].reshape(1, get_len_y*get_len_x)
            x_b_compressed = x_b[0:get_len_y, 0:get_len_x].reshape(1, get_len_y*get_len_x)
            x_r_compressed = x_r[0:get_len_y, 0:get_len_x].reshape(1, get_len_y*get_len_x)

            # 将三个通道拼接起来，记录进特征向量组
            feature_vector[k] = np.hstack([x_y_compressed, x_b_compressed, x_r_compressed])
            k += 1

    # 以行向量的方式返回
    # return feature_vector.reshape(feature_vector.shape[0] * feature_vector.shape[1])
    return feature_vector


def k_means_for_single_pic(mat, class_num, alpha=0.3):
    """
        通过 K-Means 迭代求取各个特征向量的均值分量
        :param mat: 二维矩阵，储存有经过离散余弦变换处理过的图像块信息
        :param class_num: 要形成分类的类别总数 
        :param alpha: K-Means 的学习率，决定了每次均值向量移动的步长大小 
        :return: mu: 二维矩阵，每一行都为每个分类的中心
                 Pr: 取到各个向量的概率
                 
    """
    LOOP_LIMIT = 200
    # 自动根据输入数据获取关键参数
    # num_of_pics, num_of_blocks, dimensions = mat.shape
    # num_of_lines = num_of_pics * num_of_blocks
    # mat_reshaped = mat.reshape((num_of_lines, dimensions))
    num_of_lines, dimensions = mat.shape
    mat_reshaped = mat

    max_each_class = (2 * num_of_lines / 3)

    # 初始化单张图片的数值特征猜测值
    Pr = np.ones([class_num]) / class_num                           # 概率
    mu = np.random.randn(class_num, dimensions)                     # 向量质心

    for i in range(class_num):
        mu[i] = mat_reshaped[i * int(num_of_lines/class_num)]

    # 求当前均值向量到各个块的距离
    # 此处采用刚性 k-means 分类，
    # 同时采用欧式距离衡量区别
    min_index = np.zeros([num_of_lines], dtype=np.int32)

    for loops in range(LOOP_LIMIT):                # 最多进行 50 次迭代，超时则报错 
        # print("Loop : %d" % loops)

        flags = np.ones([class_num])       # 迭代标识位

        class_count = np.zeros([class_num], dtype=np.int32)
        for i in range(num_of_lines):
            # 求取范数
            dist = np.linalg.norm(mat_reshaped[i]-mu, axis=1)
            
            # 找到范数最小值所对应的均值
            # 这种思路会导致有些类样本数为0，故不采用
            # min_index[i] = np.argwhere(dist == np.min(dist))

            judge = class_count > max_each_class
            for l in range(class_num):
                if judge[l]:
                    dist[l] = np.inf

            for l in range(class_num):
                if np.min(dist) == dist[l]:
                    min_index[i] = l

        # 提取到均值向量距离最小行的索引
        for idx in range(class_num):
            split_vector = np.int32(min_index == idx)

            split_mat =  np.tile(split_vector.reshape([num_of_lines, 1]), (1, dimensions))
            mat_splited = split_mat * mat_reshaped

            # 计算重心位移步长
            min_dist_sum = np.sum(split_vector)
            if min_dist_sum != 0:
                move = alpha * (np.sum((mat_reshaped-mu[idx])*split_mat, axis=0)/min_dist_sum)
            else:
                move = 0
            # 倘若此次位移小于阈值，则认为该方向迭代完成
            if np.linalg.norm(move) < 0.2:
                flags[idx] = 0

            # 重心移动
            mu[idx] += move

            # 修正概率值
            Pr[idx] = min_dist_sum / num_of_lines

            # print(Pr)
            # print(mu)

        print(flags)
        # 如果标识位均为 0 则认为迭代完成
        if np.sum(flags) == 0:
            print("K-Means 迭代完成")
            break

    if np.sum(flags) != 0:
        print("K-Means 未收敛")

    return mu, Pr


def k_means_extend(mu_mat, class_num, alpha=0.3):
    num_of_pics, num_of_mu, dimensions = mu_mat.shape
    num_of_lines = num_of_pics * num_of_mu

    mat = mu_mat.reshape([num_of_lines, dimensions])
    mu, Pr = k_means_for_single_pic(mat, class_num, alpha)
    return mu, Pr



def get_mat_inv(mat):
    """
        对矩阵求逆，如果矩阵的逆不存在，则
        返回伪逆
    """
    if np.linalg.det(mat) == 0:
        return np.linalg.pinv(mat)
    else:
        return np.linalg.inv(mat)


def em_for_single_pic(img, mu, Pr):
    """
        通过 EM 算法迭代求取单张图片的高斯混合模型
        :param img: 三维张量，储存有经过离散余弦变换处理过的图像块信息
        :param mu: 均值的初始猜测值
        :param Pr: 各分量在高斯混合模型中所做的贡献
        :return: new_mu: 二维矩阵，每一行都为某一高斯混合模型分量的均值
                 new_sigma: 二维矩阵，每一行都为某一高斯混合模型分量的协方差矩阵
                 new_Pr: 取到各个分量的概率
    """
    class_num, dimensions = mu.shape
    N = img.shape[0]

    # 高斯混合模型的参数
    sigma_gmm = np.zeros([class_num, dimensions, dimensions], 
                dtype=np.float64) # 协方差矩阵

    # 求取协方差矩阵
    sigma = np.zeros([class_num, dimensions, dimensions], dtype=np.float64)
    for i in range(class_num):
        sigma[i] = (np.matmul((img-mu[i]).T, (img-mu[i])))/(dimensions-1)

    # E-步：
    gamma = np.zeros([class_num, N], dtype=np.float64)
    for k in range(class_num):
        for i in range(N):
            # 求取分子
            numerator = Pr[k] * multivariate_normal(mu[k], cov=sigma[k]).pdf(img[i])

            # 求取分母
            denominator = 0
            for j in range(class_num):
                denominator += Pr[j] * multivariate_normal(mu[j], cov=sigma[j]).pdf(img[i])

            # 求得γ，E-步结束 
            gamma[k, i] = numerator / denominator


    # M-步：
    N_k = np.sum(gamma, axis=1)         # 累积分到四个类的样本数
    N_k += 1                            # 防止分母为0

    # 计算新的均值
    new_mu = np.zeros([class_num, dimensions])
    for k in range(class_num):
        tmp = np.zeros([dimensions])
        for i in range(N):
            tmp += gamma[k, i] * img[i]
        new_mu[k] = tmp / N_k[k]
        
    # 计算新的协方差矩阵
    new_sigma = np.zeros(sigma.shape)
    for k in range(class_num):
        tmp = np.zeros(sigma.shape[1:])

        for i in range(N):
            tmp += gamma[k, i] * np.matmul((img[i]-new_mu[k]).reshape([dimensions, 1]), 
                    (img[i]-new_mu[k]).reshape([1, dimensions]))

        new_sigma[k] = tmp / N_k[k]

    new_Pr = N_k / N

    # print("new_mu :", new_mu)
    # print("new_sigma :", new_sigma)
    # print("new_Pr :", new_Pr)

    return new_mu, new_sigma, new_Pr


def em_loop(img, mu, Pr):
    EM_LOOPS = 100
    mu_last = 0
    for i in range(EM_LOOPS):
        # print("Em Loop : %d" % i)
        mu, sigma, Pr = em_for_single_pic(img, mu, Pr)
        dist = np.linalg.norm(mu-mu_last)

        print(dist)
        if dist < 1e-3:
            print("EM 迭代成功")
            break

        mu_last = mu

    return mu, sigma, Pr


def norm_pdf_multivariate(x, mu, sigma):
    dimensions = x.shape[0]

    det = np.linalg.det(sigma)
    if det == 0:
        raise NameError("The covariance matrix can't be singular")

    norm_const = 1.0 / (np.power((2 * np.pi), dimensions / 2) * np.power(det, 1.0 / 2))
    x_mu = np.matrix(x - mu)
    inv = get_mat_inv(sigma)

    result = np.power(np.e, -0.5 * (x_mu * inv * x_mu.T))

    return norm_const * result


def em_for_class(mu_from_child, sigma_from_child, Pr_from_child, mu, Pr):
    """
        通过 EM 算法迭代求取一类图片的高斯混合模型
        :param mu_from_child: 经由 em 算法获得的均值向量(字密度函数参数)
        :param sigma_from_child: 经由 em 算法获得的协方差矩阵向量(字密度函数参数)
        :param Pr_from_child: 经由 em 算法获得的各模型权重(字密度函数参数)
        :param mu: 经由 K-Means 算法获得的均值初始猜测值
        :param Pr: 经由 K-Means 算法求得的各模型分量的贡献值

        :return: new_mu: 二维矩阵，每一行都为某一高斯混合模型分量的均值
                 new_sigma: 三维矩阵，每一行都为某一高斯混合模型分量的协方差矩阵
                 new_Pr: 取到各个分量的概率
    """
    num_of_pics, class_num_from_child, dimensions = mu_from_child.shape

    class_num, _ = mu.shape


    # 求取协方差矩阵
    sigma = np.zeros([class_num, dimensions, dimensions], dtype=np.float64)
    mu_mat = mu_from_child.reshape([num_of_pics*class_num_from_child, dimensions])
    for i in range(class_num):
        sigma[i] = (np.matmul((mu_mat-mu[i]).T, (mu_mat-mu[i])))/(dimensions-1)


    # E-步：
    h = np.zeros([class_num, num_of_pics, class_num_from_child], dtype=np.float64)
    for m in range(class_num):
        for j in range(num_of_pics):
            for k in range(class_num_from_child):
                # 求取分子
                numerator = Pr[m] * np.power(np.exp(-0.5 * np.trace(
                                                    get_mat_inv(sigma[m])@sigma_from_child[j, k]
                                                    ))
                            * norm_pdf_multivariate(mu_from_child[j, k], mu[m], sigma[m]),
                            Pr_from_em[j, k])


                # 求取分母
                denominator = 0
                for l in range(class_num):
                    denominator += Pr[l] * np.power(np.exp(-0.5 * np.trace(
                                                get_mat_inv(sigma[l])@sigma_from_child[j, k]
                                                ))
                            * norm_pdf_multivariate(mu_from_child[j, k], mu[l], sigma[l]),
                            Pr_from_em[j, k])
                
                h[m, j, k] = numerator / denominator

    # M步：
    # 更新各高斯模型分量权重
    new_Pr = np.zeros([class_num], dtype=np.float64)
    new_Pr = np.sum(np.sum(h, axis=2), axis=1) / (num_of_pics * class_num_from_child)

    w = np.zeros([class_num, num_of_pics, class_num_from_child], dtype=np.float64)
    for m in range(class_num):
        for j in range(num_of_pics):
            for k in range(class_num_from_child):
                numerator = h[m, j, k] * Pr_from_child[j, k]

                denominator = 0
                for l in range(num_of_pics):
                    for o in range(class_num_from_child):
                        denominator += h[m, l, o] * Pr_from_child[l, o] 

                w[m, j, k] = numerator / denominator
                # print(w[m, j, k])

    # 更新均值
    new_mu = np.zeros([class_num, dimensions], dtype=np.float64)
    for m in range(class_num):
        tmp = np.zeros([1, dimensions], dtype=np.float64)
        for j in range(num_of_pics):
            for k in range(class_num_from_child):
                tmp += w[m, j, k] * mu_from_child[j, k]

        new_mu[m] = tmp


    # 更新协方差矩阵
    new_sigma = np.zeros([class_num, dimensions, dimensions], dtype=np.float64)
    for m in range(class_num):
        for j in range(num_of_pics):
            for k in range(class_num_from_child):
                new_sigma[m] += w[m, j, k] * (sigma_from_child[j, k]
                        + ((mu_from_child[j, k] - mu[m])
                            @(mu_from_child[j, k]-mu[m]).T))


    # print("new_mu :", new_mu)
    # print("new_sigma :", new_sigma)
    # print("new_Pr :", new_Pr)

    return new_mu, new_sigma, new_Pr


def em_extended_loops(mu_from_child, sigma_from_child, Pr_from_child, mu, Pr):
    """
        用于构造扩展 EM 算法的循环
    """
    LOOPS = 100

    class_num, dimensions = mu.shape

    mu_last = np.zeros(mu.shape)
    for i in range(LOOPS):
        print("Loops %d" % i)
        mu, sigma, Pr = em_for_class(mu_from_child, sigma_from_child, Pr_from_child, mu, Pr)

        dist = np.linalg.norm(mu_last - mu)
        print("mu :", mu)
        print("sigma :", sigma)
        print("Pr :", Pr)
        print("err :", dist)

        if dist < 1e-3:
            print("聚类成功")
            break

        mu_last = mu

    return mu, sigma, Pr



if __name__ == "__main__":
    """
        主程序入口
    """
    num_of_class = 4        # 第一次聚类的类别总数
    block_size = (8, 8)     # 滑动像素块的大小
    compress = (2, 2)       # 压缩时使用的像素块大小，从左上角
    pix_move = 2            # 滑动窗口移动的步长大小
    num_of_pics = 10        # 一个类所包含的图片总数

    mu = np.zeros([num_of_pics, num_of_class, 
        compress[0]*compress[1]*3], dtype=np.float64)
    Pr = np.zeros([num_of_pics, num_of_class], dtype=np.float64)

    mu_from_em = np.zeros([num_of_pics, num_of_class,
        compress[0]*compress[1]*3], dtype=np.float64)
    sigma_from_em = np.zeros([num_of_pics, num_of_class,
        compress[0]*compress[1]*3, 
        compress[0]*compress[1]*3], dtype=np.float64)
    Pr_from_em = np.zeros([num_of_pics, num_of_class], dtype=np.float64)
    
    for i in range(num_of_pics):
        filename = "%02d.jpg" % (i+1)

        
        image = cv.imread(filename)
        img = slide_window(image, block_size, compress, pix_move)

        mu[i], Pr[i] = k_means_for_single_pic(img, num_of_class)
        print("for pic :", filename)
        print("mu :", mu[i])
        print("Pr :", Pr[i])


        mu_from_em[i], sigma_from_em[i], Pr_from_em[i] = em_loop(img, mu[i], Pr[i])
        print("mu from em :", mu_from_em[i])
        print("sigma from em :", sigma_from_em[i])
        print("Pr from em :", Pr_from_em[i])
        
        
    mu, Pr = k_means_extend(mu_from_em, 16, alpha=0.3)
    print(mu)
    print(Pr)
    
    print("=" * 50)
    print("扩展 em :")
    mu_res, sigma_res, Pr_res = em_extended_loops(mu_from_em, sigma_from_em, Pr_from_em, mu, Pr)
    print(mu_res)
    print(sigma_res)
    print(Pr_res)

