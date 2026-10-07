# method/ — 核心实验

```text
deepset/      DeepSet 参考轨（stage1 distill → stage2 verify → stage3 CE repair）
meanfield/    Mean-field 主验收轨（stage_mf + residual + formal verify）
```

每轨统一子目录：`config/` · `scripts/` · `log/` · `output/` · 代码包。

- DeepSet 代码包：`stage1` `stage2` `stage3`
- Mean-field 代码包：`stage_mf`

导入仍用包名（由 `core.paths.ensure_import_path()` 注入路径）。

详见 [`docs/ACTIVE_LAYOUT.md`](../docs/ACTIVE_LAYOUT.md)。
