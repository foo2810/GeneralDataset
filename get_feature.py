#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:58:39 2019

@author: user
"""

import sys
import os

import numpy as np
import pandas as pd
import scipy.stats as sp
from sklearn.datasets.base import Bunch

# 最頻値
def mode(series):
    mode, _ = sp.mode(series)
    return mode


# 第1四分位
def first_quartiles(series):
    return sp.scoreatpercentile(series, 25)
   
    
# 第3四分位
def third_quartiles(series):
    return sp.scoreatpercentile(series, 75)


# FFT最大
def fft_max(series):
    f = np.fft.fft(series)
    f = np.abs(f)
    return np.max(f)
  
    
# FFT最小
def fft_min(series):
    f = np.fft.fft(series)
    f = np.abs(f)
    return np.min(f)
    
    
# FFT平均
def fft_mean(series):
    f = np.fft.fft(series)
    f = np.abs(f)
    return np.mean(f)


# FFT標準偏差
def fft_std(series):
    f = np.fft.fft(series)
    f = np.abs(f)
    return np.std(f)
    

# FFT第1四分位数
def fft_first_quartiles(series):
    f = np.fft.fft(series)
    f = np.abs(f)
    return first_quartiles(f)


# FFT中央値
def fft_median(series):
    f = np.fft.fft(series)
    f = np.abs(f)
    return np.median(f)


# FFT第3四分位数
def fft_third_quartiles(series):
    f = np.fft.fft(series)
    f = np.abs(f)
    return third_quartiles(f)


sampling_freq = 50.0
## 平均スペクトル強度
def fft_mean_spectrum_intensity(series, min_freq, max_freq):
    f = np.fft.fft(series)
    freq = np.fft.fftfreq(len(series), 1/sampling_freq)
    f = np.abs(f)
    return np.mean(f[np.logical_and(min_freq < freq, freq <= max_freq)])

# 0.0-1.25Hz
def fft_000_to_125(series):
    return fft_mean_spectrum_intensity(series, 0.0, 1.25)

# 1.25-2.5Hz
def fft_125_to_250(series):
    return fft_mean_spectrum_intensity(series, 1.25, 2.50)

# 2.5-3.75Hz
def fft_250_to_375(series):
    return fft_mean_spectrum_intensity(series, 2.50, 3.75)

# 3.75-5.0Hz
def fft_375_to_500(series):
    return fft_mean_spectrum_intensity(series, 3.75, 5.0)

#5.0-6.25Hz
def fft_500_to_625(series):
    return fft_mean_spectrum_intensity(series, 5.0, 6.25)

# 6.25-7.5Hz
def fft_625_to_750(series):
    return fft_mean_spectrum_intensity(series, 6.25, 7.5)

# 7.5-8.75Hz
def fft_750_to_875(series):
    return fft_mean_spectrum_intensity(series, 7.5, 8.75)

# 8.85Hz-10.0Hz
def fft_875_to_1000(series):
    return fft_mean_spectrum_intensity(series, 8.75, 10.0)


class FeatureExtractor:
    """
    引数
        input_data: 特徴量抽出するデータ
        window_size: 抽出するウィンドウサイズ
        target_column: (複数ターゲットが含まれているデータの場合の)ターゲットカラム名
        target_label: (単数ターゲットのデータの場合の)ターゲットラベル(整数)
        features: 特徴量のリスト
        slide_width: overlappingする場合のスライド幅
        overlapping: overlappingするかどうか
    """
    @classmethod
    def get_features(cls, input_data, window_size, target_column=None, target_label=None, features=[np.min, np.max, np.mean], slide_width=0, overlapping=False, group_list=None):
        # ターゲットが1つのDataFrame内に複数存在しているとき
        if target_column is not None:
            # 目的変数を保持する
            target = input_data[target_column]
            # dataから目的変数を削除
            data = input_data.copy()
            del data[target_column]
            
            # overlappingしない
            if not overlapping:
                # 目的変数
                target = target.groupby(target.index // window_size).agg([mode])
                
                # 説明変数
                # data = data.groupby(data.index // window_size).agg([np.min, np.max, np.mean, np.var, first_quartiles, np.median, third_quartiles])
                if group_list is None:
                    data = data.groupby(data.index // window_size).agg(features)
                else:
                    print(len(data))
                    print(len(group_list))
                    data = data.groupby(group_list).agg(features)
                # マルチカラムを解消
                levels = data.columns.levels
                labels = data.columns.codes
                
                # print(levels)
                
                col_level_1 = levels[0][labels[0]]
                col_level_2 = levels[1][labels[1]]
            
                recolnames = [x + "_" + y for x, y in zip(col_level_1, col_level_2)]
                data.columns = [x.strip() for x in recolnames]
                
                return Bunch(data=data, target=pd.Series(target.values.ravel()))
            
            else:
                # ラベルが複数でoverlapping=Trueのときは、targetとdataをそのまま返す
                return Bunch(data=data, target=pd.Series(target.values.ravel()))
        
        # 1つのDataFrameで1つのラベル
        else:
            # ラベルがNoneのときはエラーで終了
            if target_label is None:
                print("Error: not input target_label", file=sys.stderr)
                sys.exit(1)
                
            else:
                data = input_data.copy()
                # overlappingなし
                if not overlapping:
                    # 説明変数
                    # data = data.groupby(data.index // window_size).agg([np.min, np.max, np.mean, np.var, first_quartiles, np.median, third_quartiles])
                    if group_list is None:
                        data = data.groupby(data.index // window_size).agg(features)
                    else:
                        data = data.groupby(group_list).agg(features)
                    
                    # マルチカラムを解消
                    levels = data.columns.levels
                    labels = data.columns.codes
                    
                    # print(levels)
                    
                    col_level_1 = levels[0][labels[0]]
                    col_level_2 = levels[1][labels[1]]
                
                    recolnames = [x + "_" + y for x, y in zip(col_level_1, col_level_2)]
                    data.columns = [x.strip() for x in recolnames]
                    
                # overlappingあり
                else:
                    # slide_widthx
                    # overlapping用に変形
                    rep_num = window_size // slide_width
                    
                    grp = data.groupby(data.index // slide_width)
                    
                    i = 0   # カウンタ
                    end = len(grp)  # 最後のグループを判定する
                    end_length = 0  # 最後のグループの長さを保持
                    rep_data = pd.DataFrame()
                    
                    for _, df in grp:
                        i+=1
                        if i == end:
                            end_length = len(df)
                        
                        for _ in range(0, rep_num):
                            rep_data = rep_data.append(df)
                        
                    print(end_length)
                    data = rep_data[slide_width:len(rep_data)-end_length]
                    
                    # overlapping用のデータが正しいかをcsvファイルで確認する
                    data.to_csv("overlapping_check.csv")
                    
                    # 特徴量抽出
                    data = data.groupby(data.index // window_size).agg(features)
                    
                    # マルチカラムを解消
                    levels = data.columns.levels
                    labels = data.columns.codes
                    
                    # print(levels)
                    
                    col_level_1 = levels[0][labels[0]]
                    col_level_2 = levels[1][labels[1]]
                
                    recolnames = [x + "_" + y for x, y in zip(col_level_1, col_level_2)]
                    data.columns = [x.strip() for x in recolnames]
                    
                    
                # 目的変量
                target = [target_label] * len(data)
                target = pd.Series(target)
                
                return Bunch(data=data, target=target)
                
            
        
        
    @classmethod
    def concat_bunch(cls, bunchs):
        data = pd.DataFrame()
        target = pd.Series()
        for b in bunchs:
            data = pd.concat([data, b.data], ignore_index=True, sort=False)
            target = pd.concat([target, pd.Series(b.target.values.ravel())], ignore_index=True, sort=False)
            
        return Bunch(data=data, target=target)
    