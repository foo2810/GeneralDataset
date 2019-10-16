#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:46:39 2019

for GeneralDataset

@author: user
"""

import sys
import pandas as pd
import numpy as np
import scipy.stats as sp
from sklearn.datasets.base import Bunch

from for_GeneralDataset import features as f
#import features as f


def _slicing(data, window_size, slide_width):
    data_len = len(data)
    i = True
    data_ = None
    
    for j in range(0, data_len, slide_width):
        if j + window_size-1 < data_len:
            df = data.loc[j:j+window_size-1]
            if i:
                data_ = df
                i = False
            else:
                data_ = pd.concat([data_, df])
                
    
    data_ = data_.reset_index(drop=True)
    
    return data_

   
def get_features(input_data, window_size, target_column=None, target_label=None, features=None, slide_width=0, overlapping=False):
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
    # target_columnとtarget_labelがNoneのときはエラー
    if target_column is None and target_label is None:
        print("Error: not input target_column or target_label", file=sys.stderr)
        sys.exit(1)
    

    if features is None:
        # Time Domain
        features_td = [np.min, np.max, np.mean, np.std, f.first_quartiles, np.median, 
                       f.third_quartiles, sp.iqr, f.abs_mean, f.abs_std, f.rms, 
                       f.frame_init, f.fram_end, f.intensity, f.skewness, f.kurtosis, f.zcr]
        # Frequency Domain - all
        features_fd = [f.power_max, f.power_max_freq, f.power_2nd, f.power_2nd_freq, f.power_std, 
                       f.power_first_quartiles, f.power_median, f.power_third_quartiles, f.power_iqr]
        # Frequency Domain - low
        features_fd_low = [f.low_power_max, f.low_power_max_freq, f.low_power_2nd, 
                           f.low_power_2nd_freq, f.low_power_std, f.low_power_first_quartiles, 
                           f.low_power_median, f.low_power_third_quartiles, f.low_power_iqr]
        # Frequency Domain - middle
        features_fd_mid = [f.mid_power_max, f.mid_power_max_freq, f.mid_power_2nd, 
                           f.mid_power_2nd_freq, f.mid_power_std, f.mid_power_first_quartiles, 
                           f.mid_power_median, f.mid_power_third_quartiles, f.mid_power_iqr]
        # Frequncy Domain - high
        features_fd_high = [f.high_power_max, f.high_power_max_freq, f.high_power_2nd, 
                            f.high_power_2nd_freq, f.high_power_std, f.high_power_first_quartiles, 
                            f.high_power_median, f.high_power_third_quartiles, f.high_power_iqr]
        
        features = features_td + features_fd + features_fd_low + features_fd_mid + features_fd_high
        
        
    if target_column is not None:
        # 目的変数を保持する
        target = input_data[target_column]
        # dataから目的変数を削除
        data = input_data.copy()
        del data[target_column]
        
        if not overlapping:
            # 目的変数
            target = target.groupby(target.index // window_size).agg([f.mode])
            
            # 説明変数
            data = data.groupby(data.index // window_size).agg(features)
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
            print("Not implement", file=sys.stderr)
            # ラベルが複数でoverlapping=Trueのときは、targetとdataをそのまま返す
            return Bunch(data=data, target=pd.Series(target.values.ravel()))
    
    
    # 1つのDataFrameで1つのラベル
    else:
        
        data = input_data.copy()
        
        if overlapping:
            if window_size <= 0:
                print("Error: window_size must be larger than 0", file=sys.stderr)
                sys.exit(1)
                
            # スライシングをする
            data = _slicing(data, window_size, slide_width)
            
#            data.to_csv("overlapping_check.csv")
            
        data_ = data.copy()
        data = data.groupby(data.index // window_size).agg(features)
        
        # マルチカラムを解消
        levels = data.columns.levels
        labels = data.columns.codes
        
        # print(levels)
        
        col_level_1 = levels[0][labels[0]]
        col_level_2 = levels[1][labels[1]]
        
        recolnames = [x + "_" + y for x, y in zip(col_level_1, col_level_2)]
        # 空白を削除
        data.columns = [x.strip() for x in recolnames]
        
        # 目的変量
        target = [target_label] * len(data)
        target = pd.Series(target)
        
        # 相関係数を結合
        data = pd.concat([data, f.get_cor(data_)], axis=1)
        
        return Bunch(data=data, target=target)
        
            
                

