# 叙事边界（最终合并包）

## 能写

- **主方法 Deploy = mf_best residual r2**（safe@32 **96.5%**、cost **0.31**）+ 同盒 M1–M3 **PROVEN_FULL**。  
- 证书定级：**局部两步 pairwise 动作安全**（声明 feat–rel · 单威胁 · 去中心化输出），**不是**多智能体闭环整局。  
- \(X_D\) = constrained subdomain（等式/耦合），非裸 44 维超矩形。  
- 平面两步 + SafeSet + Marabou → **网络依赖**（`envelope_any_a_safe=False`）。  
- **PROVEN_FULL** = 充分 SafeSet 编码下全部生成子盒 UNSAT。  
- 同协议基线：GCBF+ · MAPPO-Lag · InforMARL；HandCBF = 解析参考。  

## 不能写

- 全状态 / 全 pose / 全 threat / 无界视界已证。  
- ACTIVE 已切到 planar（**未切**）。  
- 旧 near_plus 包络与主 claim 同级。  
- BnB INCOMPLETE / pre-McCormick 有效。  
- 基线同盒必有反例；大 N / @256 已形式化。  
- pose 钉 = 策略平移不变性已证。

## 历史材料

| 材料 | 位置 |
|------|------|
| 旧包络 L2 切片 | `3_appendix/historical_envelope_mf_best_20261006/` |
| 完整旧包 | `5_frozen_sources/mf_best_20261006/` |
| planar 增量源 | `5_frozen_sources/mf_planar_netdep_20261011/` |
| 包络时代图 | `4_paper/figs/legacy/` |

## 主 claim 一句话

> 在写明的迎面 fringe 与 pose/threat 切片上，对 MF residual r2 给出平面两步 **网络依赖全覆盖形式化证书**（局部 pairwise 动作安全）；证书依赖部署网络输出，而非动作包络。
