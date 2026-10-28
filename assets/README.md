# assets

README 引用的图都放在这个目录。

| 文件名 | 用在哪 | 来源 |
| --- | --- | --- |
| `framework.png` | Architecture | 论文 Figure 2，从 arXiv HTML 提取，7283×2237 |
| `intro-panel-a.svg` | Introduction | 论文 Figure 1(a)，`intro_panel_a_error_drift.svg` |
| `intro-panel-b.svg` | Introduction | 论文 Figure 1(b)，`intro_panel_b_update_alignment.svg` |

这三张图直接从 arXiv 的 HTML 版本取得，与 PDF 一致：

```bash
curl -O https://arxiv.org/html/2610.36807v1/figures/frameworkv4.png
curl -O https://arxiv.org/html/2610.36807v1/intro_panel_a_error_drift.svg
curl -O https://arxiv.org/html/2610.36807v1/intro_panel_b_update_alignment.svg
```

补充素材（可选，暂未放入）：

- `logo.png` — 顶部横幅。若要做，建议宽度 1600–2000 px。
- 风险校准曲线、消融图 — 论文里以表格形式给出，README 用文字与表格转述即可。

替换图片时保持文件名不变，README 无需改动。
