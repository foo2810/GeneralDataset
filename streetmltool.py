#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:46:19 2019

@author: user
"""

import sensormltool as smt
import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch
import get_feature as gf
import scipy.stats as sp


def abs_mean(series):
    tmp = np.abs(series)
    return np.mean(tmp)


def abs_min(series):
    tmp = np.abs(series)
    return np.min(tmp)


def abs_max(series):
    tmp = np.abs(series)
    return np.max(tmp)


def abs_std(series):
    tmp = np.abs(series)
    return np.std(tmp)

# 二条平均平方根
def rms(series):
    return np.sqrt(np.square(series).mean(axis=0))
 

# フレーム内の初期値
def frame_init(series):
    series = series.reset_index(drop=True)
    return series[0]

# フレーム内の最終値
def fram_end(series):
    series = series.reset_index(drop=True)
    return series[len(series)-1]


# 動きの激しさ
def intensity(series):
    series = series.reset_index(drop=True)
    N = len(series)
    tmp = 0
    for i in range(N - 1):
        tmp += np.abs(series[i] - series[i + 1])
        
    tmp /= N - 1
    return tmp


# 歪度
def skewness(series):
    series = series.reset_index(drop=True)
    N = len(series)
    u = np.std(series)
    m = np.mean(series)
    tmp = 0
    for i in range(N):
        tmp += ((series[i] - m) / u) ** 3
    
    tmp /= (N - 1) * (N - 2)
    tmp *= N
    return tmp
        

# 尖度
def kurtosis(series):
    series = series.reset_index(drop=True)
    N = len(series)
    u = np.std(series)
    m = np.mean(series)
    tmp = 0
    for i in range(N):
        tmp += ((series[i] - m) / u) ** 4
        
    tmp /= (N - 1) * (N - 2) * (N - 3)
    tmp *= N * (N + 1)
    tmp += (3 * (N - 1) ** 2) / (N - 2) * (N - 3)
    return tmp
      

# ゼロ交差率
def zcr(series):
    zero_crosses = np.nonzero(np.diff(series > np.mean(series)))[0]
    return len(zero_crosses) / (len(series)-1)
    



frequency = 50.0

## 周波数パワースペクトル
def _get_power(series):
    f = np.fft.fft(series)
    abs_ = np.abs(f)
    power = abs_ ** 2
    ts = 1.0 / frequency
    freqs = np.fft.fftfreq(len(series), ts)
    idx = np.argsort(freqs)
    
    power_df = pd.Series(power[idx], index=freqs[idx])
    
    return power, power_df
    

# 最大値
def power_max(series):
    power, _ = _get_power(series)
    return max(power)

# 最大のときの周波数
def power_max_freq(series):
    _, power = _get_power(series)
    return power.idxmax()

# 2番目に大きい値
def power_2nd(series):
    power, _ = _get_power(series)
    return sorted(power)[-2]

# 2番目に大きい値の時の周波数
def power_2nd_freq(series):
    power, power_f = _get_power(series)
    second = sorted(power_f)[-2]
    tmp = power_f.index.values[power_f == second]
    return max(tmp)

# 標準偏差
def power_std(series):
    power, _ = _get_power(series)
    return np.std(power)

# 第1四分位数
def power_first_quartiles(series):
    power, _ = _get_power(series)
    return gf.first_quartiles(power)


# 中央値
def power_median(series):
    power, _ = _get_power(series)
    return np.median(power)
  
    
# 第3四分位数
def power_third_quartiles(series):
    power, _ = _get_power(series)
    return gf.third_quartiles(series)

# 四分位範囲
def power_iqr(series):
    power, _ = _get_power(series)
    return sp.iqr(power)


# 周波数帯の周波数の列を返す
def _get_power_band(series, min_frq, max_frq):
    _, power = _get_power(series)
    freq = power.index.values
    power_b = power[np.logical_and(min_frq < freq, freq <= max_frq)]
    return power_b


## 低周波数
low_min_freq = 0.0
low_max_freq = 4.2
def low_power_max(series):
    power = _get_power_band(series, low_min_freq, low_max_freq)
    return max(power)


def low_power_max_freq(series):
    power = _get_power_band(series, low_min_freq, low_max_freq)
    return power.idxmax()


def low_power_2nd(series):
    power = _get_power_band(series, low_min_freq, low_max_freq)
    return sorted(power)[-2]


def low_power_2nd_freq(series):
    power = _get_power_band(series, low_min_freq, low_max_freq)
    second = sorted(power)[-2]
    tmp = power.index.values[power == second]
    return max(tmp)


def low_power_std(series):
    power = _get_power_band(series, low_min_freq, low_max_freq)
    return np.std(power)


def low_power_first_quartiles(series):
    power = _get_power_band(series, low_min_freq, low_max_freq)
    return gf.first_quartiles(power)


def low_power_median(series):
    power = _get_power_band(series, low_min_freq, low_max_freq)
    return np.median(power)


def low_power_third_quartiles(series):
    power = _get_power_band(series, low_min_freq, low_max_freq)
    return gf.third_quartiles(power)


def low_power_iqr(series):
    power = _get_power_band(series, low_min_freq, low_max_freq)
    return sp.iqr(power)


## 中周波数
mid_min_freq = 4.2
mid_max_freq = 8.4
def mid_power_max(series):
    power = _get_power_band(series, mid_min_freq, mid_max_freq)
    return max(power)


def mid_power_max_freq(series):
    power = _get_power_band(series, mid_min_freq, mid_max_freq)
    return power.idxmax()


def mid_power_2nd(series):
    power = _get_power_band(series, mid_min_freq, mid_max_freq)
    return sorted(power)[-2]


def mid_power_2nd_freq(series):
    power = _get_power_band(series, mid_min_freq, mid_max_freq)
    second = sorted(power)[-2]
    tmp = power.index.values[power == second]
    return max(tmp)


def mid_power_std(series):
    power = _get_power_band(series, mid_min_freq, mid_max_freq)
    return np.std(power)


def mid_power_first_quartiles(series):
    power = _get_power_band(series, mid_min_freq, mid_max_freq)
    return gf.first_quartiles(power)


def mid_power_median(series):
    power = _get_power_band(series, mid_min_freq, mid_max_freq)
    return np.median(power)


def mid_power_third_quartiles(series):
    power = _get_power_band(series, mid_min_freq, mid_max_freq)
    return gf.third_quartiles(power)


def mid_power_iqr(series):
    power = _get_power_band(series, mid_min_freq, mid_max_freq)
    return sp.iqr(power)
    

## 高周波数
high_min_freq = 8.4
high_max_freq = 12.6
def high_power_max(series):
    power = _get_power_band(series, high_min_freq, high_max_freq)
    return max(power)


def high_power_max_freq(series):
    power = _get_power_band(series, high_min_freq, high_max_freq)
    return power.idxmax()


def high_power_2nd(series):
    power = _get_power_band(series, high_min_freq, high_max_freq)
    return sorted(power)[-2]


def high_power_2nd_freq(series):
    power = _get_power_band(series, high_min_freq, high_max_freq)
    second = sorted(power)[-2]
    tmp = power.index.values[power == second]
    return max(tmp)


def high_power_std(series):
    power = _get_power_band(series, high_min_freq, high_max_freq)
    return np.std(power)


def high_power_first_quartiles(series):
    power = _get_power_band(series, high_min_freq, high_max_freq)
    return gf.first_quartiles(power)


def high_power_median(series):
    power = _get_power_band(series, high_min_freq, high_max_freq)
    return np.median(power)


def high_power_third_quartiles(series):
    power = _get_power_band(series, high_min_freq, high_max_freq)
    return gf.third_quartiles(power)


def high_power_iqr(series):
    power = _get_power_band(series, high_min_freq, high_max_freq)
    return sp.iqr(power)



def data_split_x(x):
    grp = x.groupby(x.index // 256)
    
    data = None

    i = True
      
    for _, df in grp:   
        if len(df) != 256:
            len_ = 256 - len(df)
            mean_ = df.mean()
#            print(mean_)
            paddings = [[x] * len_ for x in mean_]
            
            tmp = pd.DataFrame(data={lst: padding for lst, padding in zip(df.columns, paddings)})
            df = pd.concat([df, tmp])
#            print(df)
#            print(df.shape)
            
            
#        if len(df) == 256:   
        if i:
            data = df
            i = False
        else:
            data = np.append(data, df, axis=0)

    return data


## 相関係数たち
def _cor(x, y):
    return np.corrcoef(x, y)[1, 0]    


def get_cor(input_data):
    data = data_split_x(input_data).reshape(-1, 3, 256)
    
    # 相関係数
    x_y = []
    y_z = []
    z_x = []
    # 絶対値の相関係数
    abs_x_y = []
    abs_y_z = []
    abs_z_x = []
    # パワースペクトルの相関係数
    power_x_y = []
    power_y_z = []
    power_z_x = []
    
    power_low_x_y = []
    power_low_y_z = []
    power_low_z_x = []
    
    power_mid_x_y = []
    power_mid_y_z = []
    power_mid_z_x = []
    
    power_high_x_y = []
    power_high_y_z = []
    power_high_z_x = []
    
    
    for df in data:
        # 時間領域
        x_y += [_cor(df[0], df[1])]
        y_z += [_cor(df[1], df[2])]
        z_x += [_cor(df[2], df[0])]
        abs_x_y += [_cor(abs(df[0]), abs(df[1]))]
        abs_y_z += [_cor(abs(df[1]), abs(df[2]))]
        abs_z_x += [_cor(abs(df[2]), abs(df[0]))]
        
        # 周波数領域
        power_x, _ = _get_power(df[0])
        power_y, _ = _get_power(df[1])
        power_z, _ = _get_power(df[2])

        power_x_y += [_cor(power_x, power_y)]
        power_y_z += [_cor(power_y, power_z)]
        power_z_x += [_cor(power_z, power_x)]
        
        power_x = _get_power_band(df[0], low_min_freq, low_max_freq)
        power_y = _get_power_band(df[1], low_min_freq, low_max_freq)
        power_z = _get_power_band(df[2], low_min_freq, low_max_freq)
        
        power_low_x_y += [_cor(power_x, power_y)]
        power_low_y_z += [_cor(power_y, power_z)]
        power_low_z_x += [_cor(power_z, power_x)]
        
        power_x = _get_power_band(df[0], mid_min_freq, mid_max_freq)
        power_y = _get_power_band(df[1], mid_min_freq, mid_max_freq)
        power_z = _get_power_band(df[2], mid_min_freq, mid_max_freq)
        
        power_mid_x_y += [_cor(power_x, power_y)]
        power_mid_y_z += [_cor(power_y, power_z)]
        power_mid_z_x += [_cor(power_z, power_x)]
        
        power_x = _get_power_band(df[0], high_min_freq, high_max_freq)
        power_y = _get_power_band(df[1], high_min_freq, high_max_freq)
        power_z = _get_power_band(df[2], high_min_freq, high_max_freq)
        
        power_high_x_y += [_cor(power_x, power_y)]
        power_high_y_z += [_cor(power_y, power_z)]
        power_high_z_x += [_cor(power_z, power_x)]
        

    df_cor = pd.DataFrame(
            data={'x_y': x_y, 'y_z': y_z, 'z_x': z_x, 
                  "abs_x_y": abs_x_y, "abs_y_z": abs_y_z, "abs_z_x": abs_z_x,
                  "power_x_y": power_x_y, "power_y_z": power_y_z, "power_z_x": power_z_x,
                  "power_low_x_y": power_low_x_y, "power_low_y_z": power_low_y_z, "power_low_z_x": power_low_z_x,
                  "power_mid_x_y": power_mid_x_y, "power_mid_y_z": power_mid_y_z, "power_mid_z_x": power_mid_z_x,
                  "power_high_x_y": power_high_x_y, "power_high_y_z": power_high_y_z, "power_high_z_x": power_high_z_x})
    return df_cor
        



# 汎用性を無くしたバージョン
def get_features(input_data, window_size, target_label):
    features_td = [np.min, np.max, np.mean, np.std, gf.first_quartiles, np.median, gf.third_quartiles, sp.iqr, 
                abs_mean, abs_std, rms, frame_init, fram_end, intensity, skewness, kurtosis, zcr]
    features_fd = [power_max, power_max_freq, power_2nd, power_2nd_freq, power_std, 
                   power_first_quartiles, power_median, power_third_quartiles, power_iqr]
    features_fd_low = [low_power_max, low_power_max_freq, low_power_2nd, low_power_2nd_freq, low_power_std, 
                       low_power_first_quartiles, low_power_median, low_power_third_quartiles, low_power_iqr]
    features_fd_mid = [mid_power_max, mid_power_max_freq, mid_power_2nd, mid_power_2nd_freq, mid_power_std, 
                       mid_power_first_quartiles, mid_power_median, mid_power_third_quartiles, mid_power_iqr]
    features_fd_high = [high_power_max, high_power_max_freq, high_power_2nd, high_power_2nd_freq, high_power_std, 
                       high_power_first_quartiles, high_power_median, high_power_third_quartiles, high_power_iqr]
    
    features = features_td + features_fd + features_fd_low + features_fd_mid + features_fd_high
    data = input_data.copy()
    data = data.groupby(data.index // window_size).agg(features)
    
    # マルチカラムを解消
    levels = data.columns.levels
    labels = data.columns.codes
    
    # print(levels)
    
    col_level_1 = levels[0][labels[0]]
    col_level_2 = levels[1][labels[1]]
    
    recolnames = [x + "_" + y for x, y in zip(col_level_1, col_level_2)]
    # 空白を削除p
    data.columns = [x.strip() for x in recolnames]
    
    # 目的変量
    target = [target_label] * len(data)
    target = pd.Series(target)
    
    # 相関係数を結合
    data = pd.concat([data, get_cor(input_data)], axis=1)
    
    return Bunch(data=data, target=target)



# 引数の互換性のみ
def csv_to_dataset(dirpath, filename_list, head_length_list, data_length_list, label_list, frequency_domain=False, group_list=None):
    dataset = None
    i = True

    for name, head_length, data_length, label in zip(filename_list, head_length_list, data_length_list, label_list):
        tmp = smt.csv_to_df(dirpath, name, head_cut=True, head_length=head_length, data_length=data_length)
        tmp = tmp.reset_index(drop=True)
        tmp_ds = get_features(input_data=tmp, window_size=256, target_label=label)
        if i:
            dataset = tmp_ds
            i = False
        else:
            dataset = gf.FeatureExtractor.concat_bunch([dataset, tmp_ds])
    
    return dataset


#------------------------------------------------------------------------------------#
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import LeaveOneGroupOut

import dataset_maker as dsm
import easy_cross_validation as ecv

# RandomForestの特徴量のランキングを出力する
def importance_features(clf, columns, df: pd.DataFrame, verbose=1):
    # 特徴量の重要度
    importances = clf.feature_importances_

    # 特徴量の重要度を上から順に出力する
    f = pd.DataFrame({'number': range(0, len(importances)),
                 'feature': importances[:]})
    f2 = f.sort_values('feature', ascending=False)
    f3 = f2.loc[:, 'number']

    # 特徴量の名前
    label = columns

    # 特徴量の重要度順（降順）
    indices = np.argsort(importances)[::-1]

    tmp = {}
    
    for i in range(len(importances)):
        if verbose > 0:
            print(str(i + 1) + "   " + str(label[indices[i]]) + "   " + str(importances[indices[i]]))
        tmp[label[indices[i]]] = importances[indices[i]]
        
    df = df.append(pd.Series(tmp), ignore_index=True)
    
    return df



def LOSxO_for_Ensemble(dataset, clf, clf_name, cv_name, group_list, subject_num, maker, output, output_all, visualize=False, verbose=1):
    logo = LeaveOneGroupOut()
    split = logo.split(dataset.data, dataset.target, group_list)
    
    # cross_val_LOSO()で実装していた部分    
    print(clf_name)
    predicted = np.array([])
    scores = []
    importances = pd.DataFrame()
    
    i = 1
    
    for train, test in split:
        x_train, x_test = dataset.data.iloc[train], dataset.data.iloc[test]
        y_train, y_test = dataset.target.iloc[train], dataset.target.iloc[test]
        
        clf.fit(x_train, y_train)
        pre = clf.predict(x_test)
        predicted = np.append(predicted, pre)
        # 各被験者ごとのAccuracy
        acc = accuracy_score(y_test, pre)
        scores += [acc]
        if verbose > 0:
            # 各被験者ごとのConfusion Matrix
#             cf = confusion_matrix(y_test, pre)
            cf = confusion_matrix(pre, y_test)
            print("Subject %d Accuracy = %0.4f" %(i, acc))
            print("Subject %d Confusion Matrix" %(i))
            print(cf)
            print("")
            print("Subject %d feature lanking" %(i))
            
        importances = importance_features(clf, dataset.data.columns, importances, verbose=verbose)
        print("")
        i += 1
    
    print("")
    acc_all = accuracy_score(dataset.target, predicted)
    # cf_all = confusion_matrix(dataset.target, predicted)
    cf_all = confusion_matrix(predicted, dataset.target) # trueとpredを入れ替え
    
    print("All Accuracy = %0.4f" %(acc_all))
    print("All Confusion Matrix")
    print(cf_all)
    acc_min = min(scores)
    acc_max = max(scores)
    print("Max Subject Accuracy = %0.4f" %(acc_max))
    print("Min Subject Accuracy = %0.4f" %(acc_min))

    if verbose > 1:
        print("")
        print(classification_report(dataset.target, predicted, target_names=maker.label_list))

    if visualize:
        ecv.visualizeConfusionMatrix(cf_all, maker.label_list, max_num=420, ml_name=clf_name, alog=cv_name, dirpath=maker.resultpath)
    
    return predicted, scores, importances
    
