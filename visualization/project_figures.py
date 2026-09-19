#!/usr/bin/env python3
"""Figures for the README and the project page.

Every number below is copied from the paper (arXiv:2603.07659v2): Table 1 (subset sizes),
Table 2 (DRBench accuracy), Table 7 (inference time) and the 24.68% / 7.34% statistics of
Figure 1(c). Nothing is measured or estimated here.

    python visualization/project_figures.py                  # write standalone SVGs to assets/figures/
    python visualization/project_figures.py --site site      # also refresh the inline figures of the project page

Standalone SVGs carry their own light card background so they read well on GitHub in both
light and dark mode. Inline versions use the page's CSS variables and follow its theme.
Inline figures are inserted between <!--FIG:name--> ... <!--/FIG:name--> markers (idempotent).
"""
import argparse
import os
import random
import re
from html import escape

METHODS = ["Base", "TIE", "VCD", "M3ID", "SCI3", "SCI5", "SCI7"]
BS_OVERALL = {  # Table 2, BS Subset, Overall
    "LLaVA-NeXT-8B": [18.75, 27.31, 27.89, 29.05, 32.72, 34.19, 34.92],
    "Qwen2-VL-7B": [14.52, 22.32, 23.12, 25.68, 26.94, 29.50, 31.72],
}
SCALING = {  # Table 2, Overall, rounds = 1 (base), 3, 5, 7
    "Qwen2-VL-7B": {"Bias": ["6.11", "22.74", "25.09", "27.65"], "Sensitivity": ["36.06", "43.16", "44.58", "46.54"], "BS": ["14.52", "26.94", "29.50", "31.72"]},
    "LLaVA-NeXT-8B": {"Bias": ["0.0", "23.48", "26.08", "27.01"], "Sensitivity": ["38.63", "47.20", "47.95", "47.64"], "BS": ["18.75", "32.72", "34.19", "34.92"]},
}
TEST_SIZE = 13251                                   # Sec. 4.1
BS_SIZE = {"LLaVA-NeXT": 3270, "Qwen2-VL": 1756}    # Table 1, BS Subset, Overall
SHARED_PCT = 7.34                                   # Sec. 1 / Figure 1(c)
OVERHEAD = {"SCI3": (2.96, 1.29), "SCI5": (5.01, 1.81), "SCI7": (6.68, 2.48)}  # Table 7: (sequential, batch) x base

L = {
    "en": {"bs_title": "BS Subset of DRBench · overall top-1 accuracy (%)", "ours": "SCI (ours)", "prior": "prior counterfactual decoding", "base": "base model",
           "sc_title": "Overall accuracy (%) vs. number of inference rounds", "rounds": ["1 · base", "3 · SCI3", "5 · SCI5", "7 · SCI7"],
           "Bias": "Bias", "Sensitivity": "Sensitivity", "BS": "BS",
           "ov_title": "Non-robust samples are model-specific (share of the 13,251 test samples)", "shared": "also non-robust for the other model (7.34%)", "own": "non-robust for this model only",
           "oh_title": "Inference time relative to the base model (Qwen2-VL, MMStar, one A800)", "seq": "rounds run sequentially", "batch": "batch inference", "one": "base model = 1×",
           "d_title": "One SCI5 decoding step: 5 forward passes, 1 token distribution", "orig": "Original input", "orig2": "image v⁰ + prompt q⁰",
           "tc1": "adds “focus on image details”", "tc2": "instruction language switched", "vc1": "black image, RGB (0,0,0)", "vc2": "diffusion noise, 500 steps",
           "passes": ["5 forward", "passes"], "tcn": "Textual counterfactual", "tcd": "prompt-consistent logits", "vcn": "Visual counterfactual", "vcd": "removes the language prior",
           "out": "Next token", "apc": "+ plausibility", "apc2": "constraint (β)"},
    "zh": {"bs_title": "DRBench 的 BS 子集 · 总体 top-1 准确率（%）", "ours": "SCI（本文）", "prior": "已有的反事实解码方法", "base": "基座模型",
           "sc_title": "总体准确率（%）随推理轮数的变化", "rounds": ["1 · 基座", "3 · SCI3", "5 · SCI5", "7 · SCI7"],
           "Bias": "偏见子集", "Sensitivity": "敏感性子集", "BS": "BS 子集",
           "ov_title": "非鲁棒样本因模型而异（占 13,251 个测试样本的比例）", "shared": "对另一个模型同样非鲁棒（7.34%）", "own": "仅对该模型非鲁棒",
           "oh_title": "相对基座模型的推理时间（Qwen2-VL，MMStar，单张 A800）", "seq": "各轮顺序执行", "batch": "批量推理", "one": "基座模型 = 1×",
           "d_title": "SCI5 的一步解码：5 次前向，得到 1 个 token 分布", "orig": "原始输入", "orig2": "图像 v⁰ + 提示词 q⁰",
           "tc1": "增加“关注图像细节”的指令", "tc2": "同上，并切换指令语言", "vc1": "全黑图像 RGB (0,0,0)", "vc2": "扩散加噪 500 步",
           "passes": ["5 次", "前向推理"], "tcn": "文本反事实", "tcd": "对提示词一致的 logits", "vcn": "视觉反事实", "vcd": "去除语言先验",
           "out": "下一个 token", "apc": "+ 合理性约束", "apc2": "（β）"},
}

FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',Helvetica,Arial,sans-serif"
CSS_FILE = ("text{font-family:%s;font-size:12px;fill:#59636e}.t{fill:#1f2328;font-size:14px;font-weight:600}.v{fill:#1f2328;font-weight:600;font-size:11.5px;paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round}"
            ".nm{fill:#1f2328}.h{fill:#1f2328;font-weight:600;font-size:12.5px}.m{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px;fill:#1f2328}"
            ".grid{stroke:#e6e8eb}.ax{stroke:#8c959f}.f1{fill:#2a78d6}.f2{fill:#eb6834}.f3{fill:#1baf7a}.s1{stroke:#2a78d6}.s2{stroke:#eb6834}.s3{stroke:#1baf7a}"
            ".fg{fill:#9aa4af}.fl{fill:#ced5dc}.ring{stroke:#fff;stroke-width:2}.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}"
            ".box{fill:#fff;stroke:#c9d1d9}.boxk{fill:#fff;stroke:#1f2328;stroke-width:1.5}.boxo{fill:#fdeeea;stroke:#c8341f;stroke-width:1.5}.ar{stroke:#8c959f;stroke-width:1.5;fill:none}.arh{fill:#8c959f}") % FONT
CSS_PAGE = ("text{font-family:inherit;font-size:12px;fill:var(--ink-2)}.t{fill:var(--ink);font-size:14px;font-weight:600}.v{fill:var(--ink);font-weight:600;font-size:11.5px;paint-order:stroke;stroke:var(--paper);stroke-width:3px;stroke-linejoin:round}"
            ".nm{fill:var(--ink)}.h{fill:var(--ink);font-weight:600;font-size:12.5px}.m{font-family:var(--mono);font-size:11px;fill:var(--ink)}"
            ".grid{stroke:var(--rule)}.ax{stroke:var(--ink-3)}.f1{fill:var(--s1)}.f2{fill:var(--s2)}.f3{fill:var(--s3)}.s1{stroke:var(--s1)}.s2{stroke:var(--s2)}.s3{stroke:var(--s3)}"
            ".fg{fill:var(--ink-3)}.fl{fill:var(--rule)}.ring{stroke:var(--paper);stroke-width:2}.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}"
            ".box{fill:var(--paper);stroke:var(--rule)}.boxk{fill:var(--paper);stroke:var(--ink);stroke-width:1.5}.boxo{fill:var(--pen-soft);stroke:var(--pen);stroke-width:1.5}.ar{stroke:var(--ink-3);stroke-width:1.5;fill:none}.arh{fill:var(--ink-3)}")


def wrap(name, w, h, body, aria, inline):
    if inline:  # scope the rules to this figure so they cannot leak into the page
        css = re.sub(r"(^|})([^{}]+){", lambda m: m.group(1) + ",".join(f".pf-{name} {s.strip()}" for s in m.group(2).split(",")) + "{", CSS_PAGE)
        return f'<svg class="pf-{name}" viewBox="0 0 {w} {h}" role="img" aria-label="{escape(aria)}"><style>{css}</style>{body}</svg>'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{escape(aria)}">'
            f'<style>{CSS_FILE}</style><rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="8" fill="#ffffff" stroke="#d0d7de"/>{body}</svg>')


def tx(x, y, s, cls="", anchor="start"):
    c = f' class="{cls}"' if cls else ""
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text x="{x:.1f}" y="{y:.1f}"{c}{a}>{escape(str(s))}</text>'


def fig_methods(t, inline):
    W, H, o = 760, 262, []
    o.append(tx(20, 28, t["bs_title"], "t"))
    lx = 20
    for cls, lab in (("f1", t["ours"]), ("fg", t["prior"]), ("fl", t["base"])):
        o.append(f'<rect class="{cls}" x="{lx}" y="40" width="12" height="12" rx="2"/>' + tx(lx + 17, 50.5, lab))
        lx += 30 + len(lab) * (11.5 if any(ord(ch) > 255 for ch in lab) else 6.4)
    for p, (model, vals) in enumerate(BS_OVERALL.items()):
        x0, y0, bw, sc = 20 + p * 380 + 46, 84, 250, 250 / 40
        o.append(tx(x0 - 46, y0 - 10, model, "h"))
        for g in range(0, 41, 10):
            o.append(f'<line class="grid" x1="{x0 + g * sc:.1f}" x2="{x0 + g * sc:.1f}" y1="{y0}" y2="{y0 + 7 * 22}"/>' + tx(x0 + g * sc, y0 + 7 * 22 + 15, g, anchor="middle"))
        for i, (m, v) in enumerate(zip(METHODS, vals)):
            y = y0 + i * 22
            cls = "fl" if i == 0 else ("fg" if i < 4 else "f1")
            o.append(tx(x0 - 8, y + 14, m, "nm" if i >= 4 else "", "end"))
            o.append(f'<path class="{cls}" d="M{x0},{y + 3} h{v * sc - 3:.1f} a3,3 0 0 1 3,3 v10 a3,3 0 0 1 -3,3 h-{v * sc - 3:.1f} z"/>')
            o.append(tx(x0 + v * sc + 6, y + 14.5, f"{v:.2f}", "v"))
    return wrap("methods", W, H, "".join(o), "Bar chart of overall accuracy on the BS Subset for the base model, TIE, VCD, M3ID, SCI3, SCI5 and SCI7, for LLaVA-NeXT-8B and Qwen2-VL-7B.", inline)


def fig_scaling(t, inline):
    W, H, o = 760, 300, []
    o.append(tx(20, 28, t["sc_title"], "t"))
    for p, (model, ser) in enumerate(SCALING.items()):
        L0, R0, T0, B0 = 20 + p * 375 + 30, 20 + p * 375 + 250, 66, 246
        X = lambda i: L0 + 16 + i * (R0 - L0 - 28) / 3
        Y = lambda v: B0 - float(v) / 50 * (B0 - T0)
        o.append(tx(L0 - 30, T0 - 14, model, "h"))
        for g in range(0, 51, 10):
            o.append(f'<line class="grid" x1="{L0}" x2="{R0}" y1="{Y(g):.1f}" y2="{Y(g):.1f}"/>' + tx(L0 - 7, Y(g) + 4, g, anchor="end"))
        for i, lab in enumerate(t["rounds"]):
            o.append(tx(X(i), B0 + 18, lab, anchor="middle"))
        for k, (name, vals) in enumerate(ser.items()):
            o.append(f'<polyline class="ln s{k + 1}" points="{" ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))}"/>')
        for k, (name, vals) in enumerate(ser.items()):
            for i, v in enumerate(vals):
                o.append(f'<circle class="f{k + 1} ring" cx="{X(i):.1f}" cy="{Y(v):.1f}" r="5"/>')
            o.append(tx(X(0), Y(vals[0]) - 10, vals[0], "v", "middle"))
            o.append(f'<text x="{X(3) + 11:.1f}" y="{Y(vals[3]) + 4:.1f}" class="nm">{escape(t[name])} <tspan class="v">{vals[3]}</tspan></text>')
    return wrap("scaling", W, H, "".join(o), "Line charts: overall accuracy on the Bias, Sensitivity and BS subsets rises from the base model (1 round) to SCI3, SCI5 and SCI7 for both Qwen2-VL-7B and LLaVA-NeXT-8B.", inline)


def fig_overlap(t, inline):
    W, H, o = 760, 176, []
    o.append(tx(20, 28, t["ov_title"], "t"))
    x0, sc = 118, 560 / 30
    for cls, lab, lx in (("fg", t["shared"], 20), ("f1", t["own"], 20 + 40 + len(t["shared"]) * (11.5 if any(ord(c) > 255 for c in t["shared"]) else 6.2))):
        o.append(f'<rect class="{cls}" x="{lx:.0f}" y="40" width="12" height="12" rx="2"/>' + tx(lx + 17, 50.5, lab))
    for g in range(0, 31, 5):
        o.append(f'<line class="grid" x1="{x0 + g * sc:.1f}" x2="{x0 + g * sc:.1f}" y1="68" y2="146"/>' + tx(x0 + g * sc, 162, f"{g}%", anchor="middle"))
    for r, (model, n) in enumerate(BS_SIZE.items()):
        pct, y = n / TEST_SIZE * 100, 74 + r * 38
        o.append(tx(x0 - 10, y + 17, model, "nm", "end"))
        o.append(f'<rect class="fg" x="{x0}" y="{y}" width="{SHARED_PCT * sc - 1:.1f}" height="26"/>')
        w = (pct - SHARED_PCT) * sc
        o.append(f'<path class="f1" d="M{x0 + SHARED_PCT * sc + 1:.1f},{y} h{w - 4:.1f} a3,3 0 0 1 3,3 v20 a3,3 0 0 1 -3,3 h-{w - 4:.1f} z"/>')
        o.append(tx(x0 + pct * sc + 8, y + 17, f"{pct:.2f}%  ({n:,} / {TEST_SIZE:,})", "v"))
    return wrap("overlap", W, H, "".join(o), "24.68% of the test samples are non-robust for LLaVA-NeXT and 13.25% for Qwen2-VL, but only 7.34% are shared between the two models.", inline)


def fig_overhead(t, inline):
    W, H, o = 760, 250, []
    o.append(tx(20, 28, t["oh_title"], "t"))
    lx = 20
    for cls, lab in (("fg", t["seq"]), ("f1", t["batch"])):
        o.append(f'<rect class="{cls}" x="{lx:.0f}" y="40" width="12" height="12" rx="2"/>' + tx(lx + 17, 50.5, lab))
        lx += 40 + len(lab) * (11.5 if any(ord(c) > 255 for c in lab) else 6.2)
    x0, B0, sc = 70, 212, 140 / 7
    for g in range(0, 8):
        y = B0 - g * sc
        o.append(f'<line class="grid" x1="{x0}" x2="{W - 30}" y1="{y:.1f}" y2="{y:.1f}"/>' + tx(x0 - 8, y + 4, f"{g}×", anchor="end"))
    o.append(f'<line class="ax" stroke-dasharray="4 3" x1="{x0}" x2="{W - 30}" y1="{B0 - sc:.1f}" y2="{B0 - sc:.1f}"/>' + tx(W - 34, B0 - sc - 6, t["one"], anchor="end"))
    for i, (name, (seq, bat)) in enumerate(OVERHEAD.items()):
        gx = x0 + 40 + i * 185
        for j, (val, cls) in enumerate(((seq, "fg"), (bat, "f1"))):
            bx, bh = gx + j * 58, val * sc
            o.append(f'<path class="{cls}" d="M{bx},{B0} v-{bh - 3:.1f} a3,3 0 0 1 3,-3 h44 a3,3 0 0 1 3,3 v{bh - 3:.1f} z"/>' + tx(bx + 25, B0 - bh - 7, f"{val:.2f}×", "v", "middle"))
        o.append(tx(gx + 54, B0 + 20, name, "nm", "middle"))
    return wrap("overhead", W, H, "".join(o), "Inference time relative to the base model: SCI3 2.96x sequential and 1.29x with batch inference, SCI5 5.01x and 1.81x, SCI7 6.68x and 2.48x.", inline)


def fig_diagram(t, inline):
    W, H, o = 760, 356, []
    o.append(tx(20, 28, t["d_title"], "t"))
    o.append('<defs><marker id="pfah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path class="arh" d="M0,0 L8,4 L0,8 z"/></marker></defs>')
    rnd = random.Random(7)
    cards = [("boxk", None, t["orig"], t["orig2"], "img"), ("box", "f1", "TC-V1 · q¹", t["tc1"], "q"), ("box", "f1", "TC-V2 · q²", t["tc2"], "q"),
             ("box", "f2", "VC-Color0 · v¹", t["vc1"], "black"), ("box", "f2", "VC-Noise500 · v²", t["vc2"], "noise")]
    for i, (box, edge, a, b, kind) in enumerate(cards):
        y = 48 + i * 60
        o.append(f'<rect class="{box}" x="20" y="{y}" width="246" height="50" rx="4"/>')
        if edge:
            o.append(f'<rect class="{edge}" x="20" y="{y}" width="4" height="50"/>')
        tx0, ty0 = 32, y + 9
        if kind == "img":
            o.append(f'<rect x="{tx0}" y="{ty0}" width="32" height="32" fill="#7fb3d9"/><rect x="{tx0}" y="{ty0 + 19}" width="32" height="13" fill="#5d9466"/><circle cx="{tx0 + 10}" cy="{ty0 + 11}" r="5" fill="#f2b84b"/>')
        elif kind == "black":
            o.append(f'<rect x="{tx0}" y="{ty0}" width="32" height="32" fill="#000"/>')
        elif kind == "noise":
            o.append(f'<rect x="{tx0}" y="{ty0}" width="32" height="32" fill="#8d8d8d"/>' + "".join(
                f'<rect x="{tx0 + cx * 4}" y="{ty0 + cy * 4}" width="4" height="4" fill="#{v:02x}{v:02x}{v:02x}"/>' for cx in range(8) for cy in range(8) for v in [rnd.randrange(40, 230)]))
        else:
            o.append(f'<rect class="fl" x="{tx0}" y="{ty0}" width="32" height="32" opacity="0.55"/>' + tx(tx0 + 16, ty0 + 21, "Aa", "h", "middle"))
        o.append(tx(74, y + 21, a, "h") + tx(74, y + 38, b))
        o.append(f'<path class="ar" marker-end="url(#pfah)" d="M266,{y + 25} H296"/>')
    o.append('<rect class="boxk" x="298" y="48" width="70" height="290" rx="4"/>' + tx(333, 184, "LVLM", "h", "middle"))
    for k, s in enumerate(t["passes"]):
        o.append(tx(333, 204 + k * 15, s, anchor="middle"))
    for y, cls, name, formula, note, lab in ((78, "f1", t["tcn"], "TC = max( Z⁰, Z(q¹), Z(q²) )", t["tcd"], "Z⁰  Z(q¹)  Z(q²)"), (218, "f2", t["vcn"], "VC = Z⁰ − mean( Z(v¹), Z(v²) )", t["vcd"], "Z⁰  Z(v¹)  Z(v²)")):
        o.append(f'<path class="ar" marker-end="url(#pfah)" d="M368,{y + 40} H398"/>')
        o.append(f'<rect class="boxk" x="400" y="{y}" width="232" height="82" rx="4"/><rect class="{cls}" x="400" y="{y}" width="232" height="4"/>')
        o.append(tx(412, y + 24, name, "h") + tx(412, y + 46, formula, "m") + tx(412, y + 66, note))
    o.append('<path class="ar" marker-end="url(#pfah)" d="M632,119 H694 V150"/><path class="ar" marker-end="url(#pfah)" d="M632,259 H694 V238"/>')
    o.append('<rect class="boxo" x="644" y="152" width="100" height="84" rx="4"/>' + tx(694, 172, t["out"], "h", "middle") + tx(694, 191, "TC/τ₁ + VC/τ₂", "m", "middle"))
    o.append(tx(694, 210, t["apc"], anchor="middle") + tx(694, 225, t["apc2"], anchor="middle"))
    return wrap("diagram", W, H, "".join(o), "Diagram of one SCI5 decoding step: the original input, two textual counterfactual prompts and two visual counterfactual images go through the LVLM; TC is the element-wise maximum of the original and textual counterfactual logits, VC is the original logits minus the mean of the visual counterfactual logits, and the next token is decoded from TC/tau1 + VC/tau2 under an adaptive plausibility constraint.", inline)


FIGS = {"diagram": fig_diagram, "methods": fig_methods, "scaling": fig_scaling, "overlap": fig_overlap, "overhead": fig_overhead}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="assets/figures")
    ap.add_argument("--site", help="site directory whose index.html (en) and zh/index.html get inline figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, fn in FIGS.items():
        for lang in ("en", "zh"):
            path = os.path.join(a.out, f"{name}{'' if lang == 'en' else '_zh'}.svg")
            open(path, "w", encoding="utf-8").write(fn(L[lang], False))
        print("wrote", os.path.join(a.out, name + ".svg"), "(+ _zh)")
    if a.site:
        for rel, lang in (("index.html", "en"), (os.path.join("zh", "index.html"), "zh")):
            p = os.path.join(a.site, rel)
            s = open(p, encoding="utf-8").read()
            for name, fn in FIGS.items():
                s, n = re.subn(rf"(<!--FIG:{name}-->)[\s\S]*?(<!--/FIG:{name}-->)", lambda m: m.group(1) + fn(L[lang], True) + m.group(2), s)
                if n:
                    print(f"  {rel}: refreshed {name}")
            open(p, "w", encoding="utf-8").write(s)


if __name__ == "__main__":
    main()
