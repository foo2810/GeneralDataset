#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:39:31 2019

@author: user
"""

import numpy as np
import pandas as pd
import scipy.stats as sp

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
    return first_quartiles(power)


# 中央値
def power_median(series):
    power, _ = _get_power(series)
    return np.median(power)
  
    
# 第3四分位数
def power_third_quartiles(series):
    power, _ = _get_power(series)
    return third_quartiles(series)

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
    return first_quartiles(power)


def low_power_median(series):
    power = _get_power_band(series, low_min_freq, low_max_freq)
    return np.median(power)


def low_power_third_quartiles(series):
    power = _get_power_band(series, low_min_freq, low_max_freq)
    return third_quartiles(power)


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
    return first_quartiles(power)


def mid_power_median(series):
    power = _get_power_band(series, mid_min_freq, mid_max_freq)
    return np.median(power)


def mid_power_third_quartiles(series):
    power = _get_power_band(series, mid_min_freq, mid_max_freq)
    return third_quartiles(power)


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
    return first_quartiles(power)


def high_power_median(series):
    power = _get_power_band(series, high_min_freq, high_max_freq)
    return np.median(power)


def high_power_third_quartiles(series):
    power = _get_power_band(series, high_min_freq, high_max_freq)
    return third_quartiles(power)


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
        