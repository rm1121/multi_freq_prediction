"""
时序分解统一入口：支持 SSA / FFT / Wavelet 三种方法。

设计目标：
- 作为数据预处理/变换，把每个时间序列分解为 趋势(trend) / 季节(seasonal) / 噪声(noise) / 信号(signal) 成分，用于消融实验：
  检验「预测信号到底藏在哪个时序结构里」。
- 三频金融数据（日频60天、周频12周、分时60天×8根K线），每个尺度的"原生时间单位"不同（天/周/天），
  故采用 per-scale 的 period 字典（单位=该尺度自己的时间单位）来表达截止频率，而不是 SSA 那种整数成分索引。
- 短尺度守卫：weekly 仅 12 点，FFT/小波只能给很粗的频带，硬套 cutoff 会注入人工噪声。
  T<_SHORT_T 的尺度做降级处理：
    · FFT    → 退化为「平滑 vs 残差」两段（trend=平滑, signal=原始, seasonal/noise=残差）。
    · wavelet→ 自动改用 haar（滤波器短、能多带），走正常分组即可，天生扛短序列。

分组规则（方法无关）：
    raw     = 原始序列（对照组）
    trend   = 低频/趋势（最平滑）
    seasonal= 准周期振荡（中频）
    noise   = 高频残差/噪声
    signal  = trend + seasonal（去噪后的序列）

形状约定（输入/输出同形）：
    daily   : (N,12,5,8)  -> 在 60 天轴分解
    weekly  : (N,1,12,8)  -> 在 12 周轴分解
    intraday: (N,60,13,8)  -> 在 60 天轴，把 13×8 展平为 104 通道分解
"""

import numpy as np
from scipy.fft import rfft, irfft, rfftfreq

# 延迟导入 pywt（仅 wavelet 方法需要），避免无谓依赖
try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False

from models.ssa import apply_ssa

# per-scale 默认截止周期（单位 = 该尺度自己的时间单位）
_DEFAULT_PERIODS = {
    'daily':    {'trend': 30, 'noise': 5},   # 周期>30天=趋势，<5天=噪声
    'weekly':   {'trend': 8,  'noise': 2},   # 周期>8周=趋势，<2周=噪声
    'intraday': {'trend': 30, 'noise': 5},   # 沿 60 天轴
}
# 短尺度阈值：时间轴短于此值的尺度做降级处理
_SHORT_T = 24


# ====================== 形状工具 ======================
def _reshape_to_3d(arr, scale):
    """把 (N,*,*,*) 4D 数组压成 (N, T, D)，T/D 按尺度定。"""
    N = arr.shape[0]
    if scale == 'daily':
        return arr.reshape(N, 60, 8)
    elif scale == 'weekly':
        return arr.reshape(N, 12, 8)
    elif scale == 'intraday':
        return arr.reshape(N, 60, 104)
    raise ValueError(f"Unknown scale: {scale}")


# ====================== FFT（向量化） ======================
def _fft_bandpass(arr, fs, f_low, f_high):
    """向量化 FFT 带通：arr (N,T,D) -> (N,T,D)。freqs 单位 = 1/时间单位。"""
    N, T, D = arr.shape
    freqs = rfftfreq(T, d=1.0 / fs)              # (Tf,)
    coeffs = rfft(arr, axis=1)                   # (N, Tf, D)
    mask = (freqs >= f_low) & (freqs <= f_high)
    return irfft(coeffs * mask[None, :, None], n=T, axis=1).astype(np.float32)


def _fft_group(arr, fs, group, trend_period, noise_period):
    """按 group 做 FFT 分组。arr (N,T,D)。相邻带用互斥边界，保证三带并集=全频谱。

    trend   : 低通 [0, f_t]
    seasonal: 带通 (f_t, f_n)   # 严格不等，避免与 trend/noise 边界重叠
    noise   : 高通 [f_n, fs/2]
    signal  : 去噪 [0, f_n] = trend + seasonal
    """
    f_t, f_n = 1.0 / trend_period, 1.0 / noise_period
    freqs = rfftfreq(arr.shape[1], d=1.0 / fs)
    if group == 'raw':
        return arr
    coeffs = rfft(arr, axis=1)
    if group == 'trend':
        mask = freqs <= f_t
    elif group == 'noise':
        mask = freqs >= f_n
    elif group == 'seasonal':
        mask = (freqs > f_t) & (freqs < f_n)
    elif group == 'signal':
        mask = freqs <= f_n
    else:
        raise ValueError(f"Unknown group: {group}")
    return irfft(coeffs * mask[None, :, None], n=arr.shape[1], axis=1).astype(np.float32)


# ====================== Wavelet（多分辨率） ======================
def _wavelet_group(coeffs, group, noise_levels):
    """对 pywt 系数 [cA_n, cD_n, ..., cD_1] 做分组（cD_1 最高频）。"""
    new = [np.zeros_like(c) for c in coeffs]
    n = len(coeffs)
    if group == 'raw':
        return coeffs
    if group == 'trend':
        new[0] = coeffs[0]
    elif group == 'noise':
        for k in range(1, min(noise_levels + 1, n)):
            new[k] = coeffs[k]
    elif group == 'seasonal':
        for k in range(noise_levels + 1, n):
            new[k] = coeffs[k]
    elif group == 'signal':
        new[0] = coeffs[0]
        for k in range(noise_levels + 1, n):
            new[k] = coeffs[k]
    return new


def _wavelet_group_1d(series, wavelet, level, group, noise_levels):
    if group == 'raw':
        return series
    coeffs = pywt.wavedec(series, wavelet, level=level, mode='symmetric')
    coeffs = _wavelet_group(coeffs, group, noise_levels)
    rec = pywt.waverec(coeffs, wavelet, mode='symmetric')
    return rec[:len(series)]


# ====================== 短尺度守卫 ======================
def _short_scale_fft(arr, fs, group, trend_period, noise_period):
    """短尺度 FFT 退化：trend/signal 用平滑，seasonal/noise 用残差（wiggle）。"""
    if group == 'raw':
        return arr
    smooth = _fft_bandpass(arr, fs, 0.0, 1.0 / trend_period)   # 低通=平滑
    wiggle = (arr - smooth).astype(np.float32)                 # 残差=高频
    if group == 'trend':
        return smooth.astype(np.float32)
    if group == 'signal':
        # 诚实：短尺度频带太少，无法可靠去噪，返回原始序列
        return arr.astype(np.float32)
    # seasonal / noise 都退化成 wiggle（承认二者不可分）
    return wiggle


def _short_scale_wavelet(arr, cfg):
    """短尺度用小波：自动改用 haar（滤波器短，能多带），走正常分组即可。"""
    wav = 'haar'
    T = arr.shape[1]
    lvl = min(pywt.dwt_max_level(T, wav), 3)
    nl = int(cfg.get('noise_levels', 1))
    N = arr.shape[0]
    out = np.empty_like(arr)
    for i in range(N):
        if (i + 1) % 500 == 0:
            print(f" {i+1}/{N}", end='', flush=True)
        for d in range(arr.shape[2]):
            out[i, :, d] = _wavelet_group_1d(
                arr[i, :, d], wav, lvl, cfg['group'], nl)
    return out


# ====================== 尺度入口（与 apply_ssa_to_scale 同构） ======================
def apply_fft_to_scale(arr, scale, cfg):
    group = cfg.get('group', 'raw')
    if group == 'raw':
        return arr
    p = cfg.get('periods', {}).get(scale, _DEFAULT_PERIODS[scale])
    trend_period, noise_period = p['trend'], p['noise']
    fs = 1.0  # 单位: 天/周

    arr = np.asarray(arr, dtype=np.float32)
    X = _reshape_to_3d(arr, scale)            # (N, T, D)
    T, N = X.shape[1], X.shape[0]

    if T < _SHORT_T:
        print(f"    {scale}(fft): 处理 {N} 样本（短尺度 T={T}，"
              f"退化为 2 段分解 trend=平滑/signal=原始/seasonal·noise=残差）...",
              flush=True)
        out = _short_scale_fft(X, fs, group, trend_period, noise_period)
    else:
        print(f"    {scale}(fft): 处理 {N} 样本（向量化 FFT 带通）...", flush=True)
        out = _fft_group(X, fs, group, trend_period, noise_period)
    return out.reshape(arr.shape)


def apply_wavelet_to_scale(arr, scale, cfg):
    group = cfg.get('group', 'raw')
    if group == 'raw':
        return arr
    p = cfg.get('periods', {}).get(scale, _DEFAULT_PERIODS[scale])
    trend_period, noise_period = p['trend'], p['noise']
    wav = cfg.get('wavelet', 'db4')
    lvl_cfg = cfg.get('level', None)
    nl = int(cfg.get('noise_levels', 1))

    if not _HAS_PYWT:
        raise ImportError(
            "wavelet 方法需要 PyWavelets，请运行: pip install PyWavelets")

    arr = np.asarray(arr, dtype=np.float32)
    X = _reshape_to_3d(arr, scale)
    T, N = X.shape[1], X.shape[0]

    if T < _SHORT_T:
        print(f"    {scale}(wavelet): 处理 {N} 样本（短尺度 T={T}，"
              f"自动改用 haar 多带分解）", end='', flush=True)
        out = _short_scale_wavelet(X, cfg)
        print(" 完成", flush=True)
    else:
        lvl = lvl_cfg if lvl_cfg else min(pywt.dwt_max_level(T, wav), 5)
        print(f"    {scale}(wavelet): 处理 {N} 样本（{wav}, level={lvl}）",
              end='', flush=True)
        out = np.empty_like(X)
        for i in range(N):
            if (i + 1) % 500 == 0:
                print(f" {i+1}/{N}", end='', flush=True)
            for d in range(X.shape[2]):
                out[i, :, d] = _wavelet_group_1d(
                    X[i, :, d], wav, lvl, group, nl)
        print(" 完成", flush=True)
    return out.reshape(arr.shape)


# ====================== 方法分发 ======================
def apply_fft(daily, weekly, intraday, cfg):
    scales = cfg.get('scales', ['daily', 'weekly', 'intraday'])
    if 'daily' in scales:
        daily = apply_fft_to_scale(daily, 'daily', cfg)
    if 'weekly' in scales:
        weekly = apply_fft_to_scale(weekly, 'weekly', cfg)
    if 'intraday' in scales:
        intraday = apply_fft_to_scale(intraday, 'intraday', cfg)
    return daily, weekly, intraday


def apply_wavelet(daily, weekly, intraday, cfg):
    scales = cfg.get('scales', ['daily', 'weekly', 'intraday'])
    if 'daily' in scales:
        daily = apply_wavelet_to_scale(daily, 'daily', cfg)
    if 'weekly' in scales:
        weekly = apply_wavelet_to_scale(weekly, 'weekly', cfg)
    if 'intraday' in scales:
        intraday = apply_wavelet_to_scale(intraday, 'intraday', cfg)
    return daily, weekly, intraday


def apply_decomp(daily, weekly, intraday, cfg):
    """对三频率数组统一应用分解（按 cfg['method'] 分发）。"""
    m = cfg.get('method', 'ssa')
    if m == 'ssa':
        return apply_ssa(daily, weekly, intraday, cfg)
    if m == 'fft':
        return apply_fft(daily, weekly, intraday, cfg)
    if m == 'wavelet':
        return apply_wavelet(daily, weekly, intraday, cfg)
    raise ValueError(f"Unknown method: {m}")


# ======================== 自检 ========================
if __name__ == '__main__':
    rng = np.random.RandomState(0)
    # 合成一条带趋势+周期+噪声的测试序列 (60,)
    t = np.arange(60)
    ser = (0.05 * t + np.sin(2 * np.pi * t / 12.0)
           + 0.3 * rng.randn(60)).astype(np.float32)

    # --- FFT：三个分组应可完整重建原始序列 ---
    X = ser[None, :, None]                       # (1,60,1)
    trend = _fft_group(X, 1.0, 'trend', 30, 5)
    seas = _fft_group(X, 1.0, 'seasonal', 30, 5)
    noise = _fft_group(X, 1.0, 'noise', 30, 5)
    recon = (trend + seas + noise)[0, :, 0]
    print(f"[FFT] trend+seasonal+noise 重建误差 (max abs): "
          f"{np.max(np.abs(recon - ser)):.4e}")

    # 形状保持检查（仿 daily (3,12,5,8)）
    fake_daily = rng.randn(3, 12, 5, 8).astype(np.float32)
    out = apply_fft_to_scale(fake_daily, 'daily',
                             {'group': 'signal', 'periods': _DEFAULT_PERIODS})
    print(f"[FFT] daily 形状保持: {fake_daily.shape} -> {out.shape}, "
          f"NaN={np.isnan(out).any()}")

    # --- Wavelet：需 pywt ---
    if _HAS_PYWT:
        c = pywt.wavedec(ser.astype(float), 'db4', level=3, mode='symmetric')
        rec_trend = pywt.waverec(
            _wavelet_group(c, 'trend', 1), 'db4', mode='symmetric')[:60]
        rec_seas = pywt.waverec(
            _wavelet_group(c, 'seasonal', 1), 'db4', mode='symmetric')[:60]
        rec_noise = pywt.waverec(
            _wavelet_group(c, 'noise', 1), 'db4', mode='symmetric')[:60]
        werr = np.max(np.abs((rec_trend + rec_seas + rec_noise) - ser))
        print(f"[Wavelet] trend+seasonal+noise 重建误差 (max abs): "
              f"{werr:.4e}")
        out_w = apply_wavelet_to_scale(
            fake_daily, 'daily',
            {'group': 'signal', 'wavelet': 'db4', 'noise_levels': 1,
             'periods': _DEFAULT_PERIODS})
        print(f"[Wavelet] daily 形状保持: {fake_daily.shape} -> {out_w.shape}, "
              f"NaN={np.isnan(out_w).any()}")
    else:
        print("[Wavelet] 未安装 pywt，跳过 wavelet 自检 "
              f"(pip install PyWavelets)")
