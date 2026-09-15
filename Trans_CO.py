import numpy as np
import random
from sklearn.linear_model import LassoCV,LassoLarsCV, LinearRegression
from numpy.linalg import pinv
from numpy.linalg import inv, norm
from scipy.stats import norm as stats_norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.robust.robust_linear_model import RLM
import math
import pandas as pd

import os
import shutil
import subprocess
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
import tempfile
import gc
import time

DEFAULT_SEED = 2026

def set_random_seed(seed=DEFAULT_SEED):
    """Set all Python-side random seeds used in this project."""
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed

# Deterministic default when this module is used directly.
set_random_seed(DEFAULT_SEED)

def calculate_singular_values(matrix):
    """
    计算矩阵的所有奇异值
    参数：
    matrix -- 输入的二维numpy数组

    返回：
    奇异值数组（按降序排列）
    """
    # 执行奇异值分解（SVD）
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    return s  # 返回的奇异值已按降序排列


def get_max_singular_value(matrix):
    """获取最大奇异值"""
    return np.max(calculate_singular_values(matrix))
def hard_ridge_thresholding(x, lambda_, eta):
    """混合硬阈值-岭收缩函数 Θ_hard-ridge(x; λ, η)"""
    return np.where(np.abs(x) > lambda_, x / (1 + eta), 0)
def generate_data(s=40, n=150, N=800):
    # 固定参数
    K = 5
    p = 500
    h = 6
    w = np.array([1.5, 0.75, 0, 0, -1.25])  # w = (3/2, 3/4, 0, 0, -5/4)

    # 计算中间参数
    r0 = s // 3
    s_delta = s // 5

    # 生成B矩阵
    Omega = np.random.randn(r0, K)
    U, _, _ = np.linalg.svd(Omega, full_matrices=False)
    UK = U  # 前K左奇异向量

    B = np.zeros((p, K))
    B[:r0, :] = 2 * UK  # 前r0行
    B[r0:s, :] = 0.3  # 中间s-r0行

    # 生成delta
    S = np.random.choice(np.arange(s, p), size=s_delta, replace=False)
    delta = np.zeros(p)
    delta[S] = np.random.normal(0, np.sqrt(h / s_delta), s_delta)

    # 计算目标参数beta
    beta = B @ w + delta

    # 生成源数据（K个源任务）
    X_source = []
    Y_source = []
    for k in range(K):
        X_k = np.random.normal(0, 1, (N, p))  # 协方差矩阵Σk=I
        eps_k = np.random.normal(0, 1, N)  # σk²=1
        Y_k = X_k @ B[:, k] + eps_k
        X_source.append(X_k)
        Y_source.append(Y_k)

    # 生成目标数据
    X_target = np.random.normal(0, 1, (n, p))  # 协方差矩阵Σ=I
    eps = np.random.normal(0, 1, n)  # σ²=1
    Y_target = X_target @ beta + eps

    return {
        "B": B,
        "delta": delta,
        "beta": beta,
        "X_source": X_source,
        "Y_source": Y_source,
        "X_target": X_target,
        "Y_target": Y_target
    }


def generate_data_with_outliers(K=5, s=40, n=150, N=800, o=50):
    # 固定参数
    p = 100
    h = 6
    w = np.random.uniform(low=-2.0, high=2.0, size=K)

    # 计算中间参数
    r0 = s // 3
    s_delta = s // 5

    # 生成B矩阵
    Omega = np.random.randn(r0, K)
    U, _, _ = np.linalg.svd(Omega, full_matrices=False)
    UK = U  # 前K左奇异向量

    B = np.zeros((p, K))
    B[:r0, :] = 2 * UK  # 前r0行
    B[r0:s, :] = 0.3  # 中间s-r0行

    # 生成delta
    S = np.random.choice(np.arange(s, p), size=s_delta, replace=False)
    delta = np.zeros(p)
    delta[S] = np.random.normal(10, np.sqrt(h / s_delta), s_delta)

    # 计算目标参数beta
    beta = B @ w + delta


    # 生成源数据（K个源任务）
    X_source = []
    Y_source = []
    for k in range(K):
        X_k = np.random.normal(0, 1, (N, p))  # 协方差矩阵Σk=I
        eps_k = np.random.normal(0, 1, N)  # σk²=1
        gamma = np.zeros(N)
        # 随机选择o个不同的索引位置
        random_indices = np.random.choice(N, size=int(o*N), replace=False)
        # 将这些位置的值设为outlier
        for i in random_indices:
            mean = np.random.uniform(low=1, high=20, size=1)  # 在区间 (5, 10) 上生成一个随机数
            std_dev = np.random.uniform(low=0, high=5, size=1)
            gamma[i] = np.random.normal(mean, std_dev, size=1)[0]
        Y_k = X_k @ B[:, k] + eps_k + gamma
        X_source.append(X_k)
        Y_source.append(Y_k)

    # 生成目标数据
    X_target = np.random.normal(0, 1, (n, p))  # 协方差矩阵Σ=I
    eps = np.random.normal(0, 1, n)  # σ²=1
    gamma = np.zeros(n)
    numbers = np.array(range(s, n))
    random_indices = np.random.choice(n, size=int(o * n), replace=False)

    # 将这些位置的值设为outlier
    for i in random_indices:
        mean = np.random.uniform(low=1, high=20, size=1)  # 在区间 (5, 10) 上生成一个随机数
        std_dev = np.random.uniform(low=0, high=5, size=1)
        gamma[i] = np.random.normal(mean, std_dev, size=1)[0]
    Y_target = X_target @ beta + eps + gamma

    return {
        "B": B,
        "delta": delta,
        "beta": beta,
        "X_source": X_source,
        "Y_source": Y_source,
        "X_target": X_target,
        "Y_target": Y_target,
        "outlier_index": random_indices
    }


def PTL_estimator(X_source: object,Y_source, X_target: object, y_target: object) -> object:
    # Step 1: Estimate source models and create Bb matrix
    K = len(X_source)
    p = X_target.shape[1]
    Bb = np.zeros((p, K))

    for k in range(K):
        X_k = X_source[k]
        y_k = Y_source[k]
        model = LassoLarsCV(cv=5, n_jobs=-1).fit(X_k, y_k)
        Bb[:, k] = model.coef_

    # Create projected features
    Zb = X_target @ Bb

    # Step 2: Estimate wb_ptl
    wb_ptl = pinv(Zb.T @ Zb) @ Zb.T @ y_target
    e_hat = y_target - Zb @ wb_ptl

    # Step 3: Estimate delta_ptl
    delta_model = LassoLarsCV(cv=5, n_jobs=-1).fit(X_target, e_hat)
    delta_ptl = delta_model.coef_

    # Step 4: Combine components
    beta_ptl = Bb @ wb_ptl + delta_ptl
    return beta_ptl , wb_ptl


def IPODFUN(X, Y, H, sigma, betaInit=None, method="hard", TOL=1e-3, max_iter=10000):
    N = len(Y)
    gamma = np.zeros(N)  # 保持 gamma 为一维数组
    theta = sigma * np.sqrt(2 * np.log(N))

    if betaInit is None:
        r = Y.flatten()  # 确保 r 是一维数组
        if method == "hard":
            # 使用一维索引来更新 gamma
            gamma[np.abs(r) > theta] = r[np.abs(r) > theta]
        elif method == "soft":
            gamma[r > theta] = r[r > theta] - theta
            gamma[r < -theta] = r[r < -theta] + theta
    else:
        gamma_old = gamma
        r0 = Y.flatten() - np.dot(H, Y).flatten()  # 确保 r0 和 r 一维
        niter = 0
        theta = sigma * np.sqrt(2 * np.log(N)) * np.sqrt(1 - np.diag(H))
        while True and niter < max_iter:
            niter += 1
            if niter == 1:
                r = Y.flatten() - np.dot(X, betaInit)  # r 一维
            else:
                r = np.dot(H, gamma_old).flatten() + r0  # r 一维

            if method == "hard":
                gamma = np.zeros(N)  # 保持 gamma 为一维数组
                gamma[np.abs(r) > theta] = r[np.abs(r) > theta]  # 使用一维索引
            elif method == "soft":
                gamma = np.zeros(N)
                gamma[r > theta] = r[r > theta] - theta[r > theta]
                gamma[r < -theta] = r[r < -theta] + theta[r < -theta]

            if np.max(np.abs(gamma - gamma_old)) < TOL:
                break
            else:
                gamma_old = gamma
    return {"gamma": gamma, "ress": r}

def IPODFUN_large_p(X_target: object, y_target: object, sigma):
    # Step 1: Estimate source models and create Bb matrix
    p = X_target.shape[1]
    n = len(y_target)

    X_I = np.hstack((X_target, np.eye(n)))


    beta_ipod = LassoLarsCV(cv=5, n_jobs=-1).fit(X_target, (y_target)).coef_

    gamma = y_target - X_target @ beta_ipod
    converged = False
    j = 0
    k0 = get_max_singular_value(X_I) + 1  # 默认 k0 = \|X\|_2^2 + 1
    alpha_ = 0
    loss_function = 1000000000000000
    wb_ipodtl_new = np.zeros((p, 1))
    while not converged:
        # 调整后的响应变量
        #                                            alpha_, k0)

        residual = y_target - gamma - X_target @ beta_ipod
        r_beta = beta_ipod + X_target.T @ residual / (k0 ** 2)
        alpha_ = sigma * np.sqrt(2 * np.log(n)) / (k0 ** 2)
        beta_new = theta_hard_threshold(r_beta, alpha_)

        r = y_target - X_target @ beta_new

        r_gamma = gamma + residual / (k0 ** 2)

        gamma_new = theta_hard_threshold(r_gamma, alpha_)
        #                                               alpha_, alpha_, k0)
        # 检查收敛条件（无穷范数）
        max_diff = np.max(np.abs(gamma_new - gamma))
        epsilon = 1e-3
        if max_diff < epsilon:
            converged = True
            gamma = gamma_new.copy()
        else:
            gamma = gamma_new.copy()
            beta_ipod = beta_new.copy()
        j += 1

    return {"gamma": gamma, "beta": beta_ipod, "ress": r}

def theta_hard_threshold(r, lambda_vec):
    # 应用硬阈值
    return np.where(np.abs(r) > lambda_vec, r, 0)


def theta_ipod(X, y, epsilon, lambda_vec, beta_init):
    """
    Θ-IPOD算法实现

    参数：
    X : 设计矩阵 (n×p)
    y : 响应向量 (n×1)
    epsilon : 收敛阈值
    lambda_vec : 正则化参数向量 (n×1)
    beta_init : 初始参数估计 (p×1)

    返回：
    beta_hat : 估计的参数向量
    gamma_hat : 估计的异常值向量
    """
    # 初始化变量
    n, p = X.shape
    gamma = y - X @ beta_init
    converged = False
    j = 0

    # 进行经济QR分解
    Q, R = np.linalg.qr(X, mode='economic')

    while not converged:
        # 调整后的响应变量
        y_adj = y - gamma

        # 通过回代法求解 Rβ = Q^T y_adj
        beta = np.linalg.solve(R, Q.T @ y_adj)

        # 计算残差
        r = y - X @ beta

        # 应用阈值函数更新gamma
        gamma_new = theta_hard_threshold(r, lambda_vec)

        # 检查收敛条件（无穷范数）
        max_diff = np.max(np.abs(gamma_new - gamma))
        if max_diff < epsilon:
            converged = True
        else:
            gamma = gamma_new.copy()

        j += 1

    return {"gamma": gamma, "ress": r}
# IPOD_new 函数

def IPOD(X, Y, H, method="hard", TOL=1e-3, eta=1/500):
    if X is None:
        r = 0
    else:
        r = X.shape[1]

    N = len(Y)
    ress = []
    gammas = []

    if X is None:  # X is None
        betaInit = None
        step=math.ceil((norm(Y, ord=np.inf) + 1)*eta)
        lambdas = np.arange(round(norm(Y, ord=np.inf) + 1), 0, -step)
    else:
        # Robust linear model estimation
        model = RLM(Y, X).fit()
        betaInit = model.params
        tmp = np.dot((np.eye(H.shape[0]) - H).T, Y) / np.sqrt(1 - np.diag(H))
        step = math.ceil((norm(tmp, ord=np.inf) + 1) *eta)
        lambdas = np.arange(round(norm(tmp, ord=np.inf) + 1), 0, -step)

    # 限制 lambdas 的数量，避免过多计算
    for sigma in lambdas:
        sigma = sigma / np.sqrt(2 * np.log(N))
        result = IPODFUN(X, Y, H, sigma, betaInit, method=method, TOL=TOL)
        gammas.append(result["gamma"])
        ress.append(result["ress"])
    gammas = np.column_stack(gammas)
    ress = np.column_stack(ress)

    DF = np.sum(np.abs(gammas) > 1e-5, axis=0)
    if X is not None:
        Q, _ = np.linalg.qr(X, mode='complete')
        X_unscaled = Q[:, r:].T
        Y_eff = np.dot(X_unscaled, Y.flatten())
        # 计算 sigmaSqEsts
        # 计算分子部分
        numerator = np.sum((Y_eff[:, np.newaxis] - np.dot(X_unscaled, gammas)) ** 2, axis=0)
        # 计算分母部分
        denominators = len(Y_eff) - DF
        # 将 denomiators 转换为 NumPy 数组
        denominators = np.array(denominators)

        # 找到 denomiators 中为 0 的索引
        zero_indices = np.where(denominators == 0)[0]

        # 将 sigmaSqEsts 中对应于 denomiators 中为 0 的值的元素赋值为无穷大
        sigmaSqEsts = np.divide(numerator, denominators, out=np.full_like(numerator, np.inf), where=(denominators != 0))

    else:
        sigmaSqEsts = np.sum((Y[:, np.newaxis] - gammas) ** 2, axis=0) / (len(Y) - DF)

    sigmaSqEsts[sigmaSqEsts < 0] = 0
    # Check if any column in gamma corresponds to DF greater than N - r
    if not np.all(DF < N - r):
        valid_indices = DF < N - r
        gammas = gammas[:, valid_indices]
        sigmaSqEsts = sigmaSqEsts[valid_indices]
        lambdas = lambdas[valid_indices]
        ress = ress[:, valid_indices]
        DF = DF[valid_indices]

    # Identify redundant columns in gammas
    redundant_inds = np.where(np.sum(np.abs(gammas[:, 1:] - gammas[:, :-1]), axis=0) < 0.001)[0] + 1
    if len(redundant_inds) > 0:
        gammas = np.delete(gammas, redundant_inds, axis=1)
        lambdas = np.delete(lambdas, redundant_inds)
        sigmaSqEsts = np.delete(sigmaSqEsts, redundant_inds)
        ress = np.delete(ress, redundant_inds, axis=1)
        DF = np.delete(DF, redundant_inds)

    # Identify redundant sigma square estimates
    redundant_inds = np.where(np.abs(sigmaSqEsts[1:] - sigmaSqEsts[:-1]) < 0.001)[0] + 1
    if len(redundant_inds) > 0:
        gammas = np.delete(gammas, redundant_inds, axis=1)
        lambdas = np.delete(lambdas, redundant_inds)
        sigmaSqEsts = np.delete(sigmaSqEsts, redundant_inds)
        ress = np.delete(ress, redundant_inds, axis=1)
        DF = np.delete(DF, redundant_inds)

    # Check if any column in gamma corresponds to DF greater than N/2
    if not np.all(DF < N / 3):
        valid_indices = DF < N / 3
        gammas = gammas[:, valid_indices]
        sigmaSqEsts = sigmaSqEsts[valid_indices]
        lambdas = lambdas[valid_indices]
        ress = ress[:, valid_indices]
        DF = DF[valid_indices]

    riskEst = ((N - r) * np.log(sigmaSqEsts * (N - r - DF) / (N -
                                                           r)) + (np.log(N - r) + 1) * (DF + 1)) / (N - r)

    optSet = np.where(riskEst == np.min(riskEst))[0]
    gammaOptInd = optSet[np.where(DF[optSet] == np.min(DF[optSet]))[0][0]]
    gammaOpt = np.array(gammas)[:, gammaOptInd]
    resOpt = np.array(ress)[:, gammaOptInd]
    tau = np.median(np.array(ress)[np.array(gammas)[:, gammaOptInd] == 0, gammaOptInd])
    # 计算 p 值
    resOpt_scale = ress / tau  # 计算标准化残差
    p = 2 * stats_norm.sf(np.abs(resOpt_scale))  # 计算双侧 p 值
    model = LassoLarsCV(cv=5, n_jobs=-1).fit(X, (Y - gammaOpt))
    beta = model.coef_
    return {
        "gamma": gammaOpt,
        "beta":beta,
        "sigmaSqEsts": sigmaSqEsts,
        "ress": resOpt,
        "DF": DF,
        "p": p  # 返回 p 值
    }


def IPOD_new(X, Y, H, method="hard", TOL=1e-3, eta=1/500):
    if X is None:
        r = 0
    else:
        r = X.shape[1]

    N = len(Y)
    ress = []
    gammas = []

    if X is None:  # X is None
        betaInit = None
        step=math.ceil((norm(Y, ord=np.inf) + 1)*eta)
        lambdas = np.arange(round(norm(Y, ord=np.inf) + 1), 0, -step)
    else:
        # Robust linear model estimation
        model = RLM(Y, X).fit()
        betaInit = model.params
        tmp = np.dot((np.eye(H.shape[0]) - H).T, Y) / np.sqrt(1 - np.diag(H))
        lambdas = random.randint(0, round(norm(tmp, ord=np.inf) + 1))

    sigma = lambdas / np.sqrt(2 * np.log(N))
    result = IPODFUN(X, Y, H, sigma, betaInit, method=method, TOL=TOL)
    gamma = result["gamma"]
    ress = result["ress"]
    betaOpt = np.dot(inv(np.dot(X.T, X)), np.dot(X.T, Y - gamma))
    return {
        "beta": betaOpt,
        "gamma": gamma,
        "ress": ress,
    }


def IPOD_large_p(X, Y, H, method="hard", TOL=1e-3, eta=1/500):
    if X is None:
        r = 0
    else:
        r = X.shape[1]

    N = len(Y)
    ress = []
    gammas = []
    beta = []
    if X is None:  # X is None
        betaInit = None
        step=math.ceil((norm(Y, ord=np.inf) + 1)*eta)
        lambdas = np.arange(round(norm(Y, ord=np.inf) + 1), 0, -step)
    else:
        # Robust linear model estimation
        tmp = 10000
        step = math.ceil(tmp * eta)
        lambdas = np.arange(tmp, 0, -step)

    # 限制 lambdas 的数量，避免过多计算
    for sigma in lambdas:
        sigma = sigma / np.sqrt(2 * np.log(N))
        result = IPODFUN_large_p(X, Y, sigma)
        gammas.append(result["gamma"])
        beta.append(result["beta"])
        ress.append(result["ress"])
    gammas = np.column_stack(gammas)
    beta = np.column_stack(beta)
    ress = np.column_stack(ress)

    DF = np.sum(np.abs(gammas) > 1e-5, axis=0)
    p = np.sum(np.abs(beta) > 1e-5, axis=0)
    if X is not None:
        Y_eff =  Y.flatten()
        # 计算 sigmaSqEsts
        # 计算分子部分
        numerator = np.sum((Y_eff[:, np.newaxis] - np.dot(X, beta) - gammas) ** 2, axis=0)
        # 计算分母部分
        denominators = len(Y_eff) - DF - p
        # 将 denomiators 转换为 NumPy 数组
        denominators = np.array(denominators)

        # 找到 denomiators 中为 0 的索引
        zero_indices = np.where(denominators == 0)[0]

        # 将 sigmaSqEsts 中对应于 denomiators 中为 0 的值的元素赋值为无穷大
        sigmaSqEsts = np.divide(numerator, denominators, out=np.full_like(numerator, np.inf), where=(denominators != 0))
        sigmaSqEsts = numerator
    else:
        sigmaSqEsts = np.sum((Y[:, np.newaxis] - gammas) ** 2, axis=0) / (len(Y) - DF)

    sigmaSqEsts[sigmaSqEsts < 0] = 0
    # Check if any column in gamma corresponds to DF greater than N - r

    # Identify redundant columns in gammas
    # # Identify redundant sigma square estimates

    # Check if any column in gamma corresponds to DF greater than N/2
    if not np.all(DF < N / 3):
        valid_indices = DF < N / 3
        gammas = gammas[:, valid_indices]
        beta = beta[:, valid_indices]
        sigmaSqEsts = sigmaSqEsts[valid_indices]
        lambdas = lambdas[valid_indices]
        ress = ress[:, valid_indices]
        DF = DF[valid_indices]
        p = p[valid_indices]
    if not np.all(sigmaSqEsts > 0):
        valid_indices = sigmaSqEsts > 0
        gammas = gammas[:, valid_indices]
        beta = beta[:, valid_indices]
        sigmaSqEsts = sigmaSqEsts[valid_indices]
        lambdas = lambdas[valid_indices]
        ress = ress[:, valid_indices]
        DF = DF[valid_indices]
        p = p[valid_indices]

    riskEst = (np.log(sigmaSqEsts / (N )) + (np.log(N)+1) * (DF+p+1)/N)

    optSet = np.where(riskEst == np.min(riskEst))[0]
    gammaOptInd = optSet[np.where(DF[optSet] == np.min(DF[optSet]))[0][0]]
    gammaOpt = np.array(gammas)[:, gammaOptInd]
    resOpt = np.array(ress)[:, gammaOptInd]
    betaopt = np.array(beta)[:, gammaOptInd]
    tau = np.median(np.array(ress)[np.array(gammas)[:, gammaOptInd] == 0, gammaOptInd])
    # 计算 p 值
    resOpt_scale = ress / tau  # 计算标准化残差
    p = 2 * stats_norm.sf(np.abs(resOpt_scale))  # 计算双侧 p 值
    # print(optSet, DF[optSet])

    return {
        "gamma": gammaOpt,
        "beta": betaopt,
        "sigmaSqEsts": sigmaSqEsts,
        "ress": resOpt,
        "DF": DF,
        "p": p  # 返回 p 值
    }

def loss_fun(X_target,y_target,Bb,wb_ipodtl,delta_ipodtl,gamma_new,alpha_,theta,k0):
    resid = y_target - X_target @ (Bb @ wb_ipodtl + delta_ipodtl) - gamma_new
    loss = 0.5 * np.linalg.norm(resid) ** 2
    l1_penalty = alpha_ * np.linalg.norm(delta_ipodtl, ord=1)
    gamma_penalty = k0**2*np.sum(np.where(np.abs(gamma_new) < theta,
                                    theta * np.abs(gamma_new) - 0.5 * gamma_new ** 2,
                                    0.5 * theta ** 2))
    delta_penalty = k0**2*np.sum(np.where(np.abs(delta_ipodtl) < alpha_,
                                    alpha_ * np.abs(delta_ipodtl) - 0.5 * delta_ipodtl ** 2,
                                    0.5 * alpha_ ** 2))
    loss_function_new = loss + delta_penalty + gamma_penalty
    return loss_function_new,delta_penalty

def IPODTL_estimator(Bb, X_target: object, y_target: object, sigma,TOL) -> object:
    # Step 1: Estimate source models and create Bb matrix
    p = X_target.shape[1]
    n = len(y_target)

    #     Bb[:, k] = model.coef_
        # Bb[:, k] = np.dot(inv(np.dot(X_k.T, X_k)),np.dot(X_k.T, y_k - gamma_k))
        # Bb[:, k] = beta_ols.copy()


    X_I = np.hstack((X_target,np.eye(n)))
    k0 = get_max_singular_value(X_I) + 1  # 默认 k0 = \|X\|_2^2 + 1
    Zb = X_target @ Bb
    Q, R = np.linalg.qr(Zb, mode='reduced')  # 默认m

    y_adj_ = y_target
    wb = pinv(R) @ Q.T @ y_adj_
    beta_ipodtl_new = np.zeros((p, 1))

    delta_ipodtl = LassoLarsCV(cv=5, n_jobs=-1).fit(X_target, (y_target - Zb @ wb)).coef_
    gamma = y_target - Zb @ wb - X_target @ delta_ipodtl

    converged = False
    j = 0
    alpha_ = 0
    loss_function_new3 = 1000000000000000
    while not converged:
        # 调整后的响应变量
        y_adj = y_target - gamma - X_target @ delta_ipodtl
        wb_ipodtl_new = pinv(R) @ Q.T @ y_adj
        wb_ipodtl_new = np.linalg.pinv(Zb.T @ Zb) @ Zb.T @ y_adj
        loss_function_1, delta_penalty1 = loss_fun(X_target, y_target, Bb, wb_ipodtl_new, delta_ipodtl, gamma, alpha_, alpha_,k0)
        # if loss_function_new3-loss_function_1<0 and j>1:
        #     resid_new = y_adj - Zb @ wb_ipodtl_new
        #     print(np.linalg.norm(Zb.T @ resid_new))
        #     print("1",loss_function_new3-loss_function_1,y_adj,gamma,X_target @ delta_ipodtl)

        residual = y_target - Zb @ wb_ipodtl_new - X_target @ delta_ipodtl - gamma
        r_delta = delta_ipodtl + X_target.T @ residual/(k0**2)
        alpha_ = sigma *np.sqrt(2 * np.log(n))/(k0**2)
        delta_new = theta_hard_threshold(r_delta, alpha_)
        # 更新 gamma（混合硬阈值-岭收缩）
        r = y_target - Zb @ wb_ipodtl_new - X_target @ delta_new

        r_gamma = gamma + residual / (k0**2)

        gamma_new = theta_hard_threshold(r_gamma, alpha_)
        loss_function_new3, delta_penalty3 = loss_fun(X_target, y_target, Bb, wb_ipodtl_new, delta_new, gamma_new, alpha_, alpha_,k0)
        # if loss_function_1-loss_function_new3<0 and j>1:
        #     print("3", loss_function_1,loss_function_new3,loss_function_1-loss_function_new3)
         # 检查收敛条件（无穷范数）
        max_diff_gamma = np.max(np.abs(gamma_new - gamma))/(1+np.max(np.abs(gamma)))
        max_diff_w = np.max(np.abs(wb_ipodtl_new - wb)) / (1 + np.max(np.abs(wb)))
        max_diff_delta = np.max(np.abs(delta_new - delta_ipodtl)) / (1 + np.max(np.abs(delta_ipodtl)))
        max_diff = np.max([max_diff_gamma, max_diff_w,max_diff_delta])
        epsilon = TOL
        if max_diff < epsilon or j>10000:
            converged = True
            gamma = gamma_new.copy()
            delta_ipodtl = delta_new.copy()
            wb = wb_ipodtl_new.copy()
        else:
            gamma = gamma_new.copy()
            delta_ipodtl = delta_new.copy()
            wb = wb_ipodtl_new.copy()
        j += 1
    xi = np.hstack((delta_ipodtl, gamma))

    return {"gamma": gamma,"wb":wb,"delta":delta_ipodtl, "xi":xi , "ress": r}


def IPODTL(X_source,Y_source, X, Y,  method="hard", TOL=1e-3, eta=1/500):
    method="hard"
    K = len(X_source)
    if X is None:
        r = 0
    else:
        r = X.shape[1]

    N = len(Y)
    ress = []
    gammas = []
    delta = []
    xi = []
    wb = []
    Bb = np.zeros((r, K))
    for k in range(K):
        X_k = X_source[k]
        y_k = Y_source[k]
        H_k = np.dot(np.dot(X_k, inv(np.dot(X_k.T, X_k))), X_k.T)
        result = IPOD(X_k, y_k, H_k)
        gamma_k = result["gamma"]
        model = LassoLarsCV(cv=5, n_jobs=-1).fit(X_k, y_k - gamma_k)
        Bb[:, k] = model.coef_

    if X is None:  # X is None
        betaInit = None
        step=math.ceil((norm(Y, ord=np.inf) + 1)*eta)
        lambdas = np.arange(round(norm(Y, ord=np.inf) + 1), 0, -step)
    else:
        Z = X @ Bb
        w0 = np.linalg.pinv(Z.T @ Z) @ Z.T @ Y
        Q, R = np.linalg.qr(Z, mode='reduced')  # 默认m
        w0 = pinv(R) @ Q.T @ Y
        e0 = Y - Z @ w0
        M = np.hstack([X, np.eye(len(Y))])
        k0 = get_max_singular_value(M) + 1
        delta__ = LassoLarsCV(cv=5, n_jobs=-1).fit(X, (Y - Z @ w0)).coef_
        gamma__ = Y - Z @ w0 - X @ delta__
        tmp0 = max(np.linalg.norm(delta__, ord=np.inf),np.linalg.norm(gamma__, ord=np.inf))
        tmp = k0 ** 2 *tmp0 + np.linalg.norm(M.T @ e0, ord=np.inf)
        tmp = k0 ** 2 * tmp0
        print(tmp)
        step = math.ceil(tmp * eta)
        lambdas = np.arange(tmp, 0, -step)
    # 限制 lambdas 的数量，避免过多计算

    for sigma in lambdas:
        sigma = sigma / np.sqrt(2 * np.log(N))
        result = IPODTL_estimator(Bb, X, Y, sigma,TOL)
        gammas.append(result["gamma"])
        xi.append(result["xi"])
        wb.append(result["wb"])
        delta.append(result["delta"])
        ress.append(result["ress"])
    gammas = np.column_stack(gammas)
    xi = np.column_stack(xi)
    delta = np.column_stack(delta)
    wb = np.column_stack(wb)
    ress = np.column_stack(ress)

    DF = np.sum(np.abs(xi) > 1e-5, axis=0)
    if X is not None:
        n = len(Y)
        M = np.hstack((X, np.eye(n)))
        Zb = X @ Bb
        H_z = Zb @ pinv(Zb.T @ Zb) @ Zb.T

        D, U = np.linalg.eigh(H_z)
        c = np.where(np.abs(D) < 1e-6)[0]
        U_c = U[:, c]
        C = U_c.T @ (M @ M.T) @ U_c
        D, V = np.linalg.eigh(C)
        P = V @ np.diag(1.0 / np.sqrt(D)) @ V.T @ U_c.T
        Y_eff = np.dot(P, Y.flatten())
        # 计算 sigmaSqEsts
        # 计算分子部分
        numerator = np.sum((Y_eff[:, np.newaxis] - np.dot(P, np.dot(M,xi))) ** 2, axis=0)
        # 计算分母部分
        denominators = len(Y_eff) - DF
        # 将 denomiators 转换为 NumPy 数组
        denominators = np.array(denominators)

        # 找到 denomiators 中为 0 的索引
        zero_indices = np.where(denominators == 0)[0]

        # 将 sigmaSqEsts 中对应于 denomiators 中为 0 的值的元素赋值为无穷大
        sigmaSqEsts = np.divide(numerator, denominators, out=np.full_like(numerator, np.inf), where=(denominators != 0))

    else:
        sigmaSqEsts = np.sum((Y[:, np.newaxis] - gammas) ** 2, axis=0) / (len(Y) - DF)
    sigmaSqEsts[sigmaSqEsts < 0] = 0
    # Check if any column in gamma corresponds to DF greater than N - r
    if not np.all(DF <N - r):
        valid_indices = DF < N - r
        gammas = gammas[:, valid_indices]
        wb = wb[:, valid_indices]
        delta = delta[:, valid_indices]
        sigmaSqEsts = sigmaSqEsts[valid_indices]
        lambdas = lambdas[valid_indices]
        ress = ress[:, valid_indices]
        DF = DF[valid_indices]


    # Identify redundant columns in gammas
    redundant_inds = np.where(np.sum(np.abs(gammas[:, 1:] - gammas[:, :-1]), axis=0) < 0.001)[0] + 1
    if len(redundant_inds) > 0:
        gammas = np.delete(gammas, redundant_inds, axis=1)
        wb = np.delete(wb, redundant_inds, axis=1)
        delta = np.delete(delta, redundant_inds, axis=1)
        lambdas = np.delete(lambdas, redundant_inds)
        sigmaSqEsts = np.delete(sigmaSqEsts, redundant_inds)
        ress = np.delete(ress, redundant_inds, axis=1)
        DF = np.delete(DF, redundant_inds)

    # Identify redundant sigma square estimates
    redundant_inds = np.where(np.abs(sigmaSqEsts[1:] - sigmaSqEsts[:-1]) < 0.001)[0] + 1
    if len(redundant_inds) > 0:
        gammas = np.delete(gammas, redundant_inds, axis=1)
        wb = np.delete(wb, redundant_inds, axis=1)
        delta = np.delete(delta, redundant_inds, axis=1)
        lambdas = np.delete(lambdas, redundant_inds)
        sigmaSqEsts = np.delete(sigmaSqEsts, redundant_inds)
        ress = np.delete(ress, redundant_inds, axis=1)
        DF = np.delete(DF, redundant_inds)

    # Check if any column in gamma corresponds to DF greater than N/2
    if not np.all(DF < N / 3):
        valid_indices = DF < N / 3
        gammas = gammas[:, valid_indices]
        wb = wb[:, valid_indices]
        delta = delta[:, valid_indices]
        sigmaSqEsts = sigmaSqEsts[valid_indices]
        lambdas = lambdas[valid_indices]
        ress = ress[:, valid_indices]
        DF = DF[valid_indices]

    riskEst = ((N - K) * np.log(sigmaSqEsts  / (N -K)* (N - K - DF)) + (np.log(N - K) + 1) * (DF + 1))/(N - K)
    optSet = np.where(riskEst == np.min(riskEst))[0]
    gammaOptInd = optSet[np.where(DF[optSet] == np.min(DF[optSet]))[0][0]]
    gammaOpt = np.array(gammas)[:, gammaOptInd]
    wbopt = np.array(wb)[:, gammaOptInd]
    deltaopt = np.array(delta)[:, gammaOptInd]
    resOpt = np.array(ress)[:, gammaOptInd]
    tau = np.median(np.array(ress)[np.array(gammas)[:, gammaOptInd] == 0, gammaOptInd])
    # 计算 p 值
    resOpt_scale = ress / tau  # 计算标准化残差
    p = 2 * stats_norm.sf(np.abs(resOpt_scale))  # 计算双侧 p 值

    betaOpt = Bb @ wbopt + deltaopt
    return {
        "beta": betaOpt,
        "gamma": gammaOpt,
        "sigmaSqEsts": sigmaSqEsts,
        "ress": resOpt,
        "DF": DF,
        "p": p  # 返回 p 值
    }

def IPODTL_high(X_source,Y_source, X, Y,  method="hard", TOL=1e-4, eta=1/500):
    method="hard"
    # TOL=1e-3
    K = len(X_source)
    if X is None:
        r = 0
    else:
        r = X.shape[1]

    N = len(Y)
    ress = []
    gammas = []
    delta = []
    xi = []
    wb = []
    Bb = np.zeros((r, K))
    for k in range(K):
        X_k = X_source[k]
        y_k = Y_source[k]
        H_k = np.dot(np.dot(X_k, inv(np.dot(X_k.T, X_k))), X_k.T)
        result = IPOD(X_k, y_k, H_k)
        gamma_k = result["gamma"]
        model = LassoLarsCV(cv=5, n_jobs=-1).fit(X_k, y_k - gamma_k)
        Bb[:, k] = model.coef_

    if X is None:  # X is None
        betaInit = None
        step=math.ceil((norm(Y, ord=np.inf) + 1)*eta)
        lambdas = np.arange(round(norm(Y, ord=np.inf) + 1), 0, -step)
    else:
        Z = X @ Bb
        w0 = np.linalg.pinv(Z.T @ Z) @ Z.T @ Y
        Q, R = np.linalg.qr(Z, mode='reduced')  # 默认m
        w0 = pinv(R) @ Q.T @ Y
        e0 = Y - Z @ w0
        M = np.hstack([X, np.eye(len(Y))])
        k0 = get_max_singular_value(M) + 1
        delta__ = LassoLarsCV(cv=5, n_jobs=-1).fit(X, (Y - Z @ w0)).coef_
        gamma__ = Y - Z @ w0 - X @ delta__
        tmp0 = max(np.linalg.norm(delta__, ord=np.inf),np.linalg.norm(gamma__, ord=np.inf))
        tmp = k0 ** 2 *tmp0 + np.linalg.norm(M.T @ e0, ord=np.inf)
        print(tmp)
        step = math.ceil(tmp * eta)
        lambdas = np.arange(tmp, 0, -step)
    # 限制 lambdas 的数量，避免过多计算

    for sigma in lambdas:
        sigma = sigma / np.sqrt(2 * np.log(N))
        result = IPODTL_estimator(Bb, X, Y, sigma,TOL)
        gammas.append(result["gamma"])
        xi.append(result["xi"])
        wb.append(result["wb"])
        delta.append(result["delta"])
        ress.append(result["ress"])
    gammas = np.column_stack(gammas)
    xi = np.column_stack(xi)
    delta = np.column_stack(delta)
    wb = np.column_stack(wb)
    ress = np.column_stack(ress)

    DF = np.sum(np.abs(xi) > 1e-5, axis=0)
    if X is not None:
        n = len(Y)
        M = np.hstack((X, np.eye(n)))
        Zb = X @ Bb
        H_z = Zb @ pinv(Zb.T @ Zb) @ Zb.T

        D, U = np.linalg.eigh(H_z)
        c = np.where(np.abs(D) < 1e-6)[0]
        U_c = U[:, c]
        C = U_c.T @ (M @ M.T) @ U_c
        D, V = np.linalg.eigh(C)
        P = V @ np.diag(1.0 / np.sqrt(D)) @ V.T @ U_c.T
        Y_eff = np.dot(P, Y.flatten())
        # 计算 sigmaSqEsts
        # 计算分子部分
        numerator = np.sum((Y_eff[:, np.newaxis] - np.dot(P, np.dot(M,xi))) ** 2, axis=0)
        # 计算分母部分
        denominators = len(Y_eff) - DF
        # 将 denomiators 转换为 NumPy 数组
        denominators = np.array(denominators)

        # 找到 denomiators 中为 0 的索引
        zero_indices = np.where(denominators == 0)[0]

        # 将 sigmaSqEsts 中对应于 denomiators 中为 0 的值的元素赋值为无穷大
        sigmaSqEsts = np.divide(numerator, denominators, out=np.full_like(numerator, np.inf), where=(denominators != 0))

    else:
        sigmaSqEsts = np.sum((Y[:, np.newaxis] - gammas) ** 2, axis=0) / (len(Y) - DF)
    sigmaSqEsts[sigmaSqEsts < 0] = 0
    # Check if any column in gamma corresponds to DF greater than N - r
    # # Identify redundant columns in gammas
    # # Identify redundant sigma square estimates

    # Check if any column in gamma corresponds to DF greater than N/2
    if not np.all(DF < N / 3):
        valid_indices = DF < N / 3
        gammas = gammas[:, valid_indices]
        wb = wb[:, valid_indices]
        delta = delta[:, valid_indices]
        sigmaSqEsts = sigmaSqEsts[valid_indices]
        lambdas = lambdas[valid_indices]
        ress = ress[:, valid_indices]
        DF = DF[valid_indices]

    riskEst = ((N - K) * np.log(sigmaSqEsts  / (N -K)* (N - K - DF)) + (np.log(N - K) + 1) * (DF + 1))/(N - K)
    optSet = np.where(riskEst == np.min(riskEst))[0]
    gammaOptInd = optSet[np.where(DF[optSet] == np.min(DF[optSet]))[0][0]]
    gammaOpt = np.array(gammas)[:, gammaOptInd]
    wbopt = np.array(wb)[:, gammaOptInd]
    deltaopt = np.array(delta)[:, gammaOptInd]
    resOpt = np.array(ress)[:, gammaOptInd]
    tau = np.median(np.array(ress)[np.array(gammas)[:, gammaOptInd] == 0, gammaOptInd])
    # 计算 p 值
    resOpt_scale = ress / tau  # 计算标准化残差
    p = 2 * stats_norm.sf(np.abs(resOpt_scale))  # 计算双侧 p 值

    betaOpt = Bb @ wbopt + deltaopt
    return {
        "beta": betaOpt,
        "gamma": gammaOpt,
        "sigmaSqEsts": sigmaSqEsts,
        "ress": resOpt,
        "DF": DF,
        "p": p  # 返回 p 值
    }


def score(Y_target,true_outlier, detect_index):
    TP = len(set(detect_index) & set(true_outlier))
    FP = len(set(detect_index) - set(true_outlier))
    TN = len(set(true_outlier) - set(detect_index))  # 注意：TN 的计算需要知道总样本数
    FN = len(set(true_outlier) - set(detect_index))

    # 如果知道总样本数 N，可以计算 TN：
    N = len(Y_target)  # 假设 sparse_data 是 Pandas DataFrame 或 NumPy 数组
    TN = N - (TP + FP + FN)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return TP,FP,TN,FN,f1_score

def pooled_lasso_estimator(X_source, Y_source, X_target, Y_target, random_state=1):
    X_pool = np.vstack([X_target] + X_source)
    y_pool = np.concatenate([Y_target] + Y_source)

    model = LassoCV(
        cv=5,
        fit_intercept=False,
        max_iter=50000,
        n_jobs=-1,
        random_state=random_state
    )
    model.fit(X_pool, y_pool)
    return model.coef_

def weighted_pooled_lasso_estimator(
    X_source,
    Y_source,
    X_target,
    Y_target,
    source_weight_grid=(1.5, 0.75, 0.1, 0.1, 1),
    val_size=0.2,
    random_state=1
):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_target,
        Y_target,
        test_size=val_size,
        random_state=random_state
    )

    X_src = np.vstack(X_source)
    y_src = np.concatenate(Y_source)

    best_weight = None
    best_val_mse = np.inf

    for sw in source_weight_grid:
        X_pool = np.vstack([X_tr, X_src])
        y_pool = np.concatenate([y_tr, y_src])

        obs_weight = np.concatenate([
            np.ones(len(y_tr)),
            np.full(len(y_src), sw)
        ])

        model = LassoCV(
            cv=5,
            fit_intercept=False,
            max_iter=50000,
            n_jobs=-1,
            random_state=random_state
        )
        model.fit(X_pool, y_pool, sample_weight=obs_weight)

        val_pred = X_val @ model.coef_
        val_mse = np.mean((y_val - val_pred) ** 2)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_weight = sw

    X_all = np.vstack([X_target, X_src])
    y_all = np.concatenate([Y_target, y_src])

    obs_weight_all = np.concatenate([
        np.ones(len(Y_target)),
        np.full(len(y_src), best_weight)
    ])

    final_model = LassoCV(
        cv=5,
        fit_intercept=False,
        max_iter=50000,
        n_jobs=-1,
        random_state=random_state
    )
    final_model.fit(X_all, y_all, sample_weight=obs_weight_all)

    return final_model.coef_, best_weight

def _multitask_lambda_max(X_list, y_list, task_weights=None):
    """
    Compute lambda_max for multi-task lasso.

    Objective:
        sum_t task_weight_t * ||y_t - X_t beta_t||^2 / (2 n_t)
        + lambda * sum_j ||B[:, j]||_2

    At B = 0, lambda_max = max_j ||gradient_j||_2.
    """
    T = len(X_list)
    p = X_list[0].shape[1]

    if task_weights is None:
        task_weights = np.ones(T)

    G = np.zeros((T, p))

    for t, (X_t, y_t) in enumerate(zip(X_list, y_list)):
        n_t = X_t.shape[0]
        G[t, :] = -task_weights[t] * (X_t.T @ y_t) / n_t

    lambda_max = np.max(np.linalg.norm(G, axis=0))
    return lambda_max

def compute_lipschitz_constant(X_list, task_weights=None):
    T = len(X_list)

    if task_weights is None:
        task_weights = np.ones(T)

    L = 0.0

    for t, X_t in enumerate(X_list):
        n_t = X_t.shape[0]
        smax = np.linalg.norm(X_t, ord=2)
        L_t = task_weights[t] * (smax ** 2) / n_t
        L = max(L, L_t)

    return L
def _fit_multitask_lasso(
    X_list,
    y_list,
    lam,
    B_init=None,
    task_weights=None,
    L=None,
    max_iter=10000,
    tol=1e-3
):
    T = len(X_list)
    p = X_list[0].shape[1]

    if task_weights is None:
        task_weights = np.ones(T)

    if B_init is None:
        B = np.zeros((T, p))
    else:
        B = B_init.copy()

    if L is None:
        L = compute_lipschitz_constant(X_list, task_weights)

    step = 1.0 / (L + 1e-12)

    for it in range(max_iter):
        B_old = B.copy()
        grad = np.zeros_like(B)

        for t, (X_t, y_t) in enumerate(zip(X_list, y_list)):
            n_t = X_t.shape[0]
            residual_t = X_t @ B[t, :] - y_t
            grad[t, :] = task_weights[t] * (X_t.T @ residual_t) / n_t

        V = B - step * grad

        col_norms = np.linalg.norm(V, axis=0)
        shrink = np.maximum(0.0, 1.0 - step * lam / (col_norms + 1e-12))
        B = V * shrink[np.newaxis, :]

        diff = np.linalg.norm(B - B_old, ord="fro")
        scale = max(1.0, np.linalg.norm(B_old, ord="fro"))

        if diff / scale < tol:
            break

    return B


def multitask_lasso_estimator(
    X_source,
    Y_source,
    X_target,
    Y_target,
    lambda_frac_grid=None,
    val_size=0.2,
    random_state=1,
    max_iter=10000,
    tol=1e-4
):
    """
    Multi-task Lasso baseline for target + multiple source tasks.

    It treats target and each source as one task:
        task 0 = target
        task 1,...,K = sources

    It estimates task-specific coefficients but encourages shared support
    through an l1/l2 group penalty across tasks.

    The tuning parameter is selected by target validation MSE.

    Returns:
        beta_target_hat: estimated beta for target task
        best_lambda: selected lambda
        best_val_mse: target validation MSE
    """
    if lambda_frac_grid is None:
        # Fractions of lambda_max. Larger values give sparser models.
        lambda_frac_grid = np.geomspace(1.0, 0.001, 20)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_target,
        Y_target,
        test_size=val_size,
        random_state=random_state
    )

    X_train_list = [X_tr] + X_source
    y_train_list = [y_tr] + Y_source

    T = len(X_train_list)

    # Equal task weights: each domain contributes comparably,
    # instead of letting large source sample sizes dominate.
    task_weights = np.ones(T)

    lambda_max = _multitask_lambda_max(
        X_train_list,
        y_train_list,
        task_weights=task_weights
    )

    if lambda_max <= 0 or not np.isfinite(lambda_max):
        return np.zeros(X_target.shape[1]), np.nan, np.nan

    lambda_grid = lambda_max * np.asarray(lambda_frac_grid)

    best_lambda = None
    best_val_mse = np.inf
    best_B = None

    B_warm = None

    # Warm start from large lambda to small lambda.
    for lam in lambda_grid:
        B_hat = _fit_multitask_lasso(
            X_train_list,
            y_train_list,
            lam=lam,
            B_init=B_warm,
            task_weights=task_weights,
            max_iter=max_iter,
            tol=tol
        )

        B_warm = B_hat

        beta_target_tmp = B_hat[0, :]
        val_pred = X_val @ beta_target_tmp
        val_mse = np.mean((y_val - val_pred) ** 2)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_lambda = lam
            best_B = B_hat.copy()

    # Refit on full target + source using selected lambda.
    X_full_list = [X_target] + X_source
    y_full_list = [Y_target] + Y_source

    B_final = _fit_multitask_lasso(
        X_full_list,
        y_full_list,
        lam=best_lambda,
        B_init=best_B,
        task_weights=np.ones(len(X_full_list)),
        max_iter=max_iter,
        tol=tol
    )

    beta_target_hat = B_final[0, :]

    return beta_target_hat, best_lambda, best_val_mse

def save_replication_for_r(data_dir, X_source, Y_source, X_target, Y_target):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(data_dir / "X_target.csv", X_target, delimiter=",")
    np.savetxt(data_dir / "Y_target.csv", Y_target, delimiter=",")

    np.savetxt(data_dir / "K.txt", np.array([len(X_source)]), fmt="%d")

    for k, (Xk, yk) in enumerate(zip(X_source, Y_source), start=1):
        np.savetxt(data_dir / f"X_source_{k}.csv", Xk, delimiter=",")
        np.savetxt(data_dir / f"Y_source_{k}.csv", yk, delimiter=",")

def run_r_script(script_path, data_dir, out_dir,seed):
    """
    Run one R baseline script.

    This version does not print R output during successful runs.
    If R fails, it prints stdout/stderr to help debugging.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # cmd = ["/usr/local/conda3/envs/R-env/bin/Rscript", str(script_path), str(data_dir), str(out_dir),str(seed)]
    cmd = [r"C:\Program Files\R\R-4.4.2\bin\Rscript.exe", str(script_path), str(data_dir), str(out_dir),str(seed)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    import os

    cmd = ["/usr/local/conda3/envs/R-env/bin/Rscript", str(script_path), str(data_dir), str(out_dir),str(seed)]
    env = os.environ.copy()
    env["PATH"] = (
            "/usr/local/conda3/envs/R-env/bin:"
            + env.get("PATH", "")
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"R script failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


def read_r_beta(out_dir, filename):
    path = Path(out_dir) / filename
    if not path.exists():
        return None
    beta = pd.read_csv(path, header=None).values.flatten()
    return beta


def read_r_gamma(out_dir, filename):
    path = Path(out_dir) / filename
    if not path.exists():
        return None
    gamma = pd.read_csv(path, header=None).values.flatten()
    return gamma

def run_r_baselines_temp(X_source, Y_source, X_target, Y_target,seed):
    """
    Run Trans-Lasso, Sparse LTS, and Trans-PtLR using temporary folders.

    The temporary CSV files are automatically deleted after this function finishes.
    Therefore, tmp_r_data/ and tmp_r_results/ will not accumulate thousands of files.
    """

    beta_hat_translasso = None
    beta_hat_sparse_lts = None
    outlier_hat_sparse_lts = None
    beta_hat_transptlr = None

    time_translasso = np.nan
    time_sparse_lts = np.nan
    time_transptlr = np.nan

    with tempfile.TemporaryDirectory(prefix="transipod_r_data_") as data_dir, \
         tempfile.TemporaryDirectory(prefix="transipod_r_results_") as out_dir:

        # Save current replication only temporarily
        save_replication_for_r(
            data_dir,
            X_source,
            Y_source,
            X_target,
            Y_target
        )

        # ---------- Trans-Lasso ----------
        t0 = time.perf_counter()

        run_r_script(
            "R_baselines/run_translasso.R",
            data_dir,
            out_dir,
            seed=seed
        )

        beta_hat_translasso = read_r_beta(out_dir, "translasso_beta.csv")

        time_translasso = time.perf_counter() - t0


        # ---------- Sparse LTS ----------
        t0 = time.perf_counter()

        run_r_script(
            "R_baselines/run_sparse_lts.R",
            data_dir,
            out_dir,
            seed=seed
        )

        beta_hat_sparse_lts = read_r_beta(out_dir, "sparse_lts_beta.csv")
        outlier_hat_sparse_lts = read_r_gamma(out_dir, "sparse_lts_outlier_hat.csv")

        time_sparse_lts = time.perf_counter() - t0


        # ---------- Trans-PtLR ----------
        t0 = time.perf_counter()

        run_r_script(
            "R_baselines/run_transptlr.R",
            data_dir,
            out_dir,
            seed=seed
        )

        beta_hat_transptlr = read_r_beta(out_dir, "transptlr_beta.csv")

        time_transptlr = time.perf_counter() - t0

    # Once the with-block ends, data_dir and out_dir are automatically deleted.
    return {
        "beta_hat_translasso": beta_hat_translasso,
        "beta_hat_sparse_lts": beta_hat_sparse_lts,
        "outlier_hat_sparse_lts": outlier_hat_sparse_lts,
        "beta_hat_transptlr": beta_hat_transptlr,

        "time_translasso": time_translasso,
        "time_sparse_lts": time_sparse_lts,
        "time_transptlr": time_transptlr
    }

B=50