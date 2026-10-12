# 合并日志 · mf_final_20261012

| 字段 | 值 |
|------|-----|
| 日期 | 2026-10-12 |
| 类型 | 统一最终验收包 |
| 主源 | `5_frozen_sources/mf_planar_netdep_20261011/`（**完整文件复制**，无硬链接） |
| 并入 | `mf_best_20261006/2_baselines/` |
| 历史切片 | `3_appendix/historical_envelope_*`（L2 near_plus） |
| 论文 | `4_paper/{part,figs,aamas_2027_template}` |
| ACTIVE | 未改 |
| 源包 | 冻结保留 |

## 原则

1. 主 claim = planar netdep。  
2. **禁止**对源包使用硬链接改写（避免污染冻结源）。  
3. L3 超大 CE 等不整包搬入；需要时打开 `mf_best_20261006`。

## 2026-10-12 清理

- 将 `mf_best_20261006` 与 `mf_planar_netdep_20261011` **完整 rsync** 入 `5_frozen_sources/` 后，从 `acceptance/` 根目录删除。
- 主证据树 / 基线 / 论文仍在包根；归档仅供追溯（含 L3 CE、r1/midF 等原先未切片进主树的文件）。
- 清理前 spot-check：L3 CE、r1 pkl、midF pkl、planar r2+M3 REPORT 哈希一致。

## 2026-10-12 文档归类整理

- 主文迁入 `0_docs/`；根目录证书/EVAL 等改为软链
- 删除重复：`mf_best_r2_reference/` 任务 JSON、r3 双 pkl、historical r2 pkl、latency 快照目录、aamas 重复 pdf（→ 链到 figs/final）
- GAP / CHECKPOINT / MERGE 入 `3_appendix/process_notes/`
- 空 `docs/` 删除；historical docs 改为指向 `5_frozen_sources`

## 2026-10-12 symlink 消除

- 现象：`git add` 报 `stat(...repair_best.pkl): Function not implemented`（`core.symlinks=false` 常见于 WSL/Windows 混用）。
- 处理：包内全部符号链接改为普通文件；根目录 `part`/`figs`/`aamas_2027_template` 改为含 README 的指针目录（不整树复制）。
- r3 `repair_best.pkl` 现为 `repair_manual_pick.pkl` 的实体副本。
- 本机已设 `core.symlinks=true`；包本身不再依赖 symlink 以便跨环境 add/commit。

## 2026-10-12 入库范围收敛

- **只提交** final 主树；`5_frozen_sources/` 与 AAMAS LaTeX **不入库**（gitignore）。
- 删除包内 AAMAS 目录；根目录重复 md / 指针目录清理；`figs` 仅 PDF；r3 单 pkl。
- MANIFEST/SHA 仅覆盖可提交文件集。
