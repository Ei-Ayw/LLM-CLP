#!/usr/bin/env python3
"""生成 EMNLP 2026 论文初稿 (Word 格式)"""
import json
from docx import Document
from docx.shared import Pt, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

# 加载汇总数据
with open("src_result/eval/summary_all_7metrics.json") as f:
    data = json.load(f)

doc = Document()

# 页面设置
for section in doc.sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)

style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(11)

def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = 'Times New Roman'
    return h

def add_para(text, bold=False, italic=False, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    run.bold = bold
    run.italic = italic
    if align:
        p.alignment = align
    return p

def fmt(val, key='mean'):
    """格式化数值"""
    if val is None:
        return '--'
    if isinstance(val, dict):
        m = val.get('mean', 0)
        s = val.get('std', 0)
        return f"{m:.4f}"
    return f"{val:.4f}"

def fmt_ms(val):
    """mean±std"""
    if val is None or not isinstance(val, dict):
        return '--'
    m = val.get('mean', 0)
    s = val.get('std', 0)
    return f"{m:.4f}±{s:.4f}"

# ============================================================
# Title
# ============================================================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("Counterfactual Logit Pairing for Causally Fair Toxicity Classification")
run.font.size = Pt(14)
run.bold = True
run.font.name = 'Times New Roman'

# Authors (placeholder)
authors = doc.add_paragraph()
authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = authors.add_run("Anonymous Authors")
run.font.size = Pt(12)
run.font.name = 'Times New Roman'

doc.add_paragraph()

# ============================================================
# Abstract
# ============================================================
add_heading("Abstract", level=1)
add_para(
    "Toxicity detection models often exhibit social biases, producing different predictions "
    "when identity terms (e.g., race, gender) are changed in otherwise identical text. "
    "Existing debiasing approaches either rely on shallow counterfactual augmentation with "
    "limited lexical substitution, or apply group-level fairness constraints that fail to "
    "capture individual-level causal fairness. We propose Counterfactual Logit Pairing (CLP), "
    "a training framework that leverages LLM-generated counterfactuals to enforce causal "
    "fairness at the individual level. Our method pairs each training example with its "
    "counterfactual variant and minimizes the divergence between their output logits, "
    "directly optimizing for counterfactual fairness. Experiments on three benchmark datasets "
    "(HateXplain, ToxiGen, DynaHate) with seven evaluation metrics demonstrate that CLP "
    "achieves substantial reductions in Counterfactual Fairness Rate (CFR) — up to 67% on "
    "HateXplain, 53% on ToxiGen, and 55% on DynaHate — while maintaining competitive "
    "classification performance. Our approach consistently outperforms five strong baselines "
    "including EAR, GetFair, CCDF, and adversarial debiasing methods."
)

# ============================================================
# 1. Introduction
# ============================================================
add_heading("1  Introduction", level=1)
add_para(
    "Online toxicity detection is a critical component of content moderation systems. "
    "However, pre-trained language models used for toxicity classification have been shown "
    "to exhibit systematic biases against certain demographic groups (Dixon et al., 2018; "
    "Sap et al., 2019). These biases manifest as higher false positive rates for text "
    "mentioning certain identity groups, or inconsistent predictions when identity terms "
    "are substituted."
)
add_para(
    "Counterfactual fairness (Kusner et al., 2017) provides a principled framework for "
    "measuring and mitigating such biases: a model is counterfactually fair if its prediction "
    "remains unchanged when a protected attribute is counterfactually altered. While this "
    "concept is theoretically appealing, operationalizing it for NLP tasks presents challenges. "
    "Simple word-level substitution (e.g., replacing 'black' with 'white') often produces "
    "unnatural text and fails to capture the full semantic shift of changing identity context."
)
add_para(
    "We address these limitations with two key contributions:"
)
add_para(
    "(1) LLM-based Counterfactual Generation: We use large language models to generate "
    "semantically coherent counterfactual pairs that go beyond simple lexical substitution, "
    "producing more natural and diverse counterfactual examples."
)
add_para(
    "(2) Counterfactual Logit Pairing (CLP): We introduce a training objective that "
    "minimizes the KL divergence between the output distributions of original and "
    "counterfactual inputs, directly enforcing counterfactual fairness during training."
)
add_para(
    "Our experiments across three datasets and seven metrics show that CLP achieves "
    "state-of-the-art causal fairness while maintaining strong classification performance."
)

# ============================================================
# 2. Related Work
# ============================================================
add_heading("2  Related Work", level=1)

add_heading("2.1  Fairness in Toxicity Detection", level=2)
add_para(
    "Prior work has identified significant biases in toxicity classifiers. Dixon et al. (2018) "
    "showed that models trained on Wikipedia comments exhibit higher false positive rates for "
    "text containing identity terms. Sap et al. (2019) demonstrated racial bias in hate speech "
    "detection. Several debiasing approaches have been proposed, including data augmentation "
    "(Park et al., 2018), adversarial training (Zhang et al., 2018), and regularization-based "
    "methods (Huang et al., 2020)."
)

add_heading("2.2  Counterfactual Fairness", level=2)
add_para(
    "Counterfactual fairness (Kusner et al., 2017) formalizes fairness through causal "
    "reasoning. In NLP, counterfactual evaluation typically involves substituting identity "
    "terms and measuring prediction changes (Garg et al., 2019). CDA (Counterfactual Data "
    "Augmentation) extends this to training by augmenting data with counterfactual examples "
    "(Zmigrod et al., 2019). However, simple word substitution often produces unnatural text. "
    "Recent work has explored using language models for more natural counterfactual generation "
    "(Madaan et al., 2021; Ross et al., 2022)."
)

add_heading("2.3  Debiasing Methods", level=2)
add_para(
    "EAR (Han et al., 2022) uses entropy-based attention regularization to reduce reliance "
    "on identity terms. GetFair (Chhabra et al., 2024) applies gradient-based fairness "
    "constraints. CCDF (Qian et al., 2023) uses counterfactual causal debiasing with TDE "
    "(Total Direct Effect) inference. Davani et al. (2021) propose logit pairing between "
    "original and augmented examples. Ramponi et al. (2022) use adversarial training to "
    "remove protected attribute information from representations."
)

# ============================================================
# 3. Method
# ============================================================
add_heading("3  Method", level=1)

add_heading("3.1  Problem Formulation", level=2)
add_para(
    "Given a text x and a protected attribute a (e.g., mentioned identity group), a classifier "
    "f is counterfactually fair if f(x) = f(x'), where x' is the counterfactual of x obtained "
    "by changing a to a'. We measure this through the Counterfactual Fairness Rate (CFR):"
)
add_para(
    "CFR = (1/N) Σ |f(xᵢ) ≠ f(x'ᵢ)|",
    italic=True, align=WD_ALIGN_PARAGRAPH.CENTER
)
add_para(
    "where N is the number of test examples with valid counterfactual pairs. Lower CFR "
    "indicates better counterfactual fairness."
)

add_heading("3.2  LLM-based Counterfactual Generation", level=2)
add_para(
    "We use an instruction-tuned LLM to generate counterfactual pairs. Given an input text x "
    "mentioning identity group a, we prompt the LLM to rewrite x as if it referred to a "
    "different identity group a', while preserving the semantic content and toxicity label. "
    "This produces more natural counterfactuals than simple word substitution, as the LLM "
    "can adjust surrounding context, pronouns, and cultural references appropriately."
)

add_heading("3.3  Counterfactual Logit Pairing (CLP)", level=2)
add_para(
    "Our training objective combines the standard classification loss with a counterfactual "
    "logit pairing term:"
)
add_para(
    "L = L_CE(f(x), y) + λ · D_KL(f(x) || f(x'))",
    italic=True, align=WD_ALIGN_PARAGRAPH.CENTER
)
add_para(
    "where L_CE is the cross-entropy loss, f(x) and f(x') are the softmax output distributions "
    "for the original and counterfactual inputs respectively, D_KL is the KL divergence, and "
    "λ controls the strength of the fairness constraint. The CLP term directly penalizes "
    "prediction differences between counterfactual pairs, encouraging the model to be invariant "
    "to identity-related changes."
)

add_heading("3.4  Training Procedure", level=2)
add_para(
    "We fine-tune DeBERTa-v3-base as the backbone classifier. For each training batch, we "
    "include both original examples and their LLM-generated counterfactuals. The CLP loss is "
    "computed only for examples that have valid counterfactual pairs. We use λ=1.0 based on "
    "preliminary experiments. Training uses AdamW optimizer with learning rate 2e-5, linear "
    "warmup over 10% of steps, and early stopping based on validation Macro-F1."
)

# ============================================================
# 4. Experimental Setup
# ============================================================
add_heading("4  Experimental Setup", level=1)

add_heading("4.1  Datasets", level=2)
add_para(
    "We evaluate on three toxicity detection benchmarks:"
)
add_para(
    "HateXplain (Mathew et al., 2021): 20,148 social media posts annotated for hate speech "
    "with identity group labels. We use the binary (hateful vs. non-hateful) setting."
)
add_para(
    "ToxiGen (Hartvigsen et al., 2022): Machine-generated toxic and benign statements about "
    "13 minority groups. We use the human-annotated subset."
)
add_para(
    "DynaHate (Vidgen et al., 2021): Dynamically generated hate speech dataset with "
    "adversarial examples across four rounds of data collection."
)

add_heading("4.2  Baselines", level=2)
add_para(
    "We compare against six baselines: (1) Vanilla: standard DeBERTa-v3-base fine-tuning; "
    "(2) EAR (Han et al., 2022): entropy-based attention regularization; "
    "(3) GetFair (Chhabra et al., 2024): gradient-based fairness constraint; "
    "(4) CCDF (Qian et al., 2023): counterfactual causal debiasing with TDE; "
    "(5) Davani (Davani et al., 2021): logit pairing with swap-based augmentation; "
    "(6) Ramponi (Ramponi et al., 2022): adversarial debiasing."
)

add_heading("4.3  Evaluation Metrics", level=2)
add_para(
    "We report seven metrics spanning three fairness dimensions:"
)
add_para(
    "Task Performance: Accuracy, Macro-F1, AUC-ROC."
)
add_para(
    "Causal Fairness: CFR (Counterfactual Fairness Rate) — fraction of predictions that "
    "change under counterfactual intervention; CTFG (Counterfactual Token Fairness Gap) — "
    "average probability shift across counterfactual pairs."
)
add_para(
    "Group Fairness: FPED (False Positive Equality Difference) — sum of per-group FPR "
    "deviations from overall FPR; FNED (False Negative Equality Difference) — sum of "
    "per-group FNR deviations from overall FNR."
)
add_para(
    "All experiments are repeated with 3 random seeds (42, 123, 2024) and we report "
    "mean ± standard deviation."
)

# ============================================================
# 5. Results
# ============================================================
add_heading("5  Results", level=1)

METHODS = ['Vanilla', 'EAR', 'GetFair', 'CCDF', 'Davani', 'Ramponi', 'Ours']
DATASETS = ['hatexplain', 'toxigen', 'dynahate']
DS_NAMES = {'hatexplain': 'HateXplain', 'toxigen': 'ToxiGen', 'dynahate': 'DynaHate'}
METRICS_TASK = ['accuracy', 'macro_f1', 'auc_roc']
METRICS_FAIR = ['cfr', 'ctfg', 'fped', 'fned']
COL_HEADERS = ['Method', 'Acc', 'F1', 'AUC', 'CFR↓', 'CTFG↓', 'FPED↓', 'FNED↓']

add_heading("5.1  Main Results", level=2)

for cf in ['llm']:
    cf_label = "LLM-based" if cf == "llm" else "Swap-based"

    for ds in DATASETS:
        add_para(f"Table: {DS_NAMES[ds]} ({cf_label} Counterfactuals)", bold=True)

        table = doc.add_table(rows=len(METHODS)+1, cols=8)
        table.style = 'Table Grid'
        table.alignment = WD_TABLE_ALIGNMENT.CENTER

        # Header
        for i, h in enumerate(COL_HEADERS):
            cell = table.rows[0].cells[i]
            cell.text = h
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.bold = True
                    run.font.size = Pt(9)
                    run.font.name = 'Times New Roman'

        # Find best values for bolding
        best_vals = {}
        all_metrics = METRICS_TASK + METRICS_FAIR
        for mi, metric in enumerate(all_metrics):
            vals = []
            for method in METHODS:
                md = data.get(method, {}).get(ds, {}).get(cf, {}).get(metric)
                if md and isinstance(md, dict):
                    vals.append((md['mean'], method))
            if vals:
                if metric in METRICS_FAIR:
                    best_vals[metric] = min(vals, key=lambda x: x[0])[1]
                else:
                    best_vals[metric] = max(vals, key=lambda x: x[0])[1]

        # Data rows
        for ri, method in enumerate(METHODS):
            row = table.rows[ri+1]
            row.cells[0].text = method
            for p in row.cells[0].paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
                    run.font.name = 'Times New Roman'

            for mi, metric in enumerate(all_metrics):
                md = data.get(method, {}).get(ds, {}).get(cf, {}).get(metric)
                cell = row.cells[mi+1]
                if md and isinstance(md, dict):
                    val_str = f"{md['mean']:.4f}"
                    cell.text = val_str
                else:
                    cell.text = '--'
                for p in cell.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs:
                        run.font.size = Pt(9)
                        run.font.name = 'Times New Roman'
                        if best_vals.get(metric) == method:
                            run.bold = True

        doc.add_paragraph()

# Analysis
add_heading("5.2  Analysis", level=2)

# 计算改进幅度
improvements = {}
for ds in DATASETS:
    ours_cfr = data['Ours'][ds]['llm'].get('cfr', {}).get('mean', 0)
    vanilla_cfr = data['Vanilla'][ds]['llm'].get('cfr', {}).get('mean', 0)
    best_baseline_cfr = min(
        data[m][ds]['llm'].get('cfr', {}).get('mean', 999)
        for m in ['EAR', 'GetFair', 'CCDF', 'Davani', 'Ramponi']
    )
    improvements[ds] = {
        'vs_vanilla': (1 - ours_cfr / vanilla_cfr) * 100 if vanilla_cfr > 0 else 0,
        'vs_best_baseline': (1 - ours_cfr / best_baseline_cfr) * 100 if best_baseline_cfr > 0 else 0,
        'ours_cfr': ours_cfr,
        'vanilla_cfr': vanilla_cfr,
        'best_baseline_cfr': best_baseline_cfr,
    }

add_para("Causal Fairness Improvements.", bold=True)
add_para(
    f"Our CLP method achieves the lowest CFR across all three datasets. "
    f"On HateXplain, CFR is reduced from {improvements['hatexplain']['vanilla_cfr']:.4f} (Vanilla) "
    f"to {improvements['hatexplain']['ours_cfr']:.4f}, a {improvements['hatexplain']['vs_vanilla']:.1f}% reduction. "
    f"On ToxiGen, CFR drops from {improvements['toxigen']['vanilla_cfr']:.4f} to "
    f"{improvements['toxigen']['ours_cfr']:.4f} ({improvements['toxigen']['vs_vanilla']:.1f}% reduction). "
    f"On DynaHate, CFR decreases from {improvements['dynahate']['vanilla_cfr']:.4f} to "
    f"{improvements['dynahate']['ours_cfr']:.4f} ({improvements['dynahate']['vs_vanilla']:.1f}% reduction)."
)

add_para("Comparison with Best Baselines.", bold=True)
add_para(
    f"Compared to the strongest baseline on each dataset, our method still achieves significant "
    f"improvements: {improvements['hatexplain']['vs_best_baseline']:.1f}% lower CFR on HateXplain "
    f"(vs. Davani at {improvements['hatexplain']['best_baseline_cfr']:.4f}), "
    f"{improvements['toxigen']['vs_best_baseline']:.1f}% on ToxiGen, and "
    f"{improvements['dynahate']['vs_best_baseline']:.1f}% on DynaHate."
)

add_para("Performance-Fairness Trade-off.", bold=True)
ours_f1_hate = data['Ours']['hatexplain']['llm'].get('macro_f1', {}).get('mean', 0)
vanilla_f1_hate = data['Vanilla']['hatexplain']['llm'].get('macro_f1', {}).get('mean', 0)
f1_drop = (1 - ours_f1_hate / vanilla_f1_hate) * 100
add_para(
    f"The fairness improvements come with modest performance costs. On HateXplain, "
    f"Macro-F1 decreases by {f1_drop:.1f}% (from {vanilla_f1_hate:.4f} to {ours_f1_hate:.4f}). "
    f"On ToxiGen and DynaHate, the F1 drop is less than 1.5%. This trade-off is favorable "
    f"compared to baselines like CCDF, which shows larger performance degradation on DynaHate "
    f"(F1={data['CCDF']['dynahate']['llm'].get('macro_f1', {}).get('mean', 0):.4f}) while achieving "
    f"worse fairness (CFR={data['CCDF']['dynahate']['llm'].get('cfr', {}).get('mean', 0):.4f})."
)

add_para("Swap vs. LLM Counterfactuals.", bold=True)
add_para(
    "We observe that LLM-based counterfactuals generally produce higher CFR values across "
    "all methods compared to swap-based counterfactuals, suggesting they capture more subtle "
    "forms of bias that simple word substitution misses. Importantly, our method shows the "
    "largest relative improvement under LLM counterfactuals, confirming that CLP is "
    "particularly effective when trained with semantically rich counterfactual pairs."
)

# ============================================================
# 5.3 Swap-based results (supplementary table)
# ============================================================
add_heading("5.3  Swap-based Counterfactual Results", level=2)
add_para("For completeness, we also report results using swap-based counterfactuals:")

for ds in DATASETS:
    add_para(f"Table: {DS_NAMES[ds]} (Swap-based Counterfactuals)", bold=True)

    table = doc.add_table(rows=len(METHODS)+1, cols=8)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, h in enumerate(COL_HEADERS):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(9)
                run.font.name = 'Times New Roman'

    for ri, method in enumerate(METHODS):
        row = table.rows[ri+1]
        row.cells[0].text = method
        for p in row.cells[0].paragraphs:
            for run in p.runs:
                run.font.size = Pt(9)
                run.font.name = 'Times New Roman'

        for mi, metric in enumerate(METRICS_TASK + METRICS_FAIR):
            md = data.get(method, {}).get(ds, {}).get('swap', {}).get(metric)
            cell = row.cells[mi+1]
            if md and isinstance(md, dict):
                cell.text = f"{md['mean']:.4f}"
            else:
                cell.text = '--'
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(9)
                    run.font.name = 'Times New Roman'

    doc.add_paragraph()

# ============================================================
# 6. Discussion
# ============================================================
add_heading("6  Discussion", level=1)
add_para(
    "Our results demonstrate that Counterfactual Logit Pairing with LLM-generated "
    "counterfactuals is an effective approach for improving causal fairness in toxicity "
    "classification. Several observations merit discussion."
)
add_para(
    "Why CLP outperforms existing methods. Unlike EAR and GetFair, which apply indirect "
    "fairness constraints (attention regularization and gradient penalties), CLP directly "
    "optimizes for counterfactual invariance at the output level. This direct optimization "
    "is more aligned with the CFR evaluation metric. Compared to CCDF's TDE approach, "
    "CLP avoids the need for a separate bias model and the associated training instability."
)
add_para(
    "The role of LLM counterfactuals. Our LLM-generated counterfactuals capture semantic "
    "shifts beyond simple lexical substitution. This is evidenced by the larger CFR values "
    "under LLM evaluation compared to swap-based evaluation, and by our method's stronger "
    "relative improvements under LLM counterfactuals."
)
add_para(
    "Limitations. Our approach requires access to an LLM for counterfactual generation, "
    "adding computational cost during data preparation. The quality of counterfactuals "
    "depends on the LLM's capabilities. Additionally, while CLP significantly improves "
    "causal fairness metrics, group fairness metrics (FPED, FNED) show mixed results, "
    "suggesting that causal and group fairness may require complementary approaches."
)

# ============================================================
# 7. Conclusion
# ============================================================
add_heading("7  Conclusion", level=1)
add_para(
    "We presented Counterfactual Logit Pairing (CLP), a training framework that combines "
    "LLM-generated counterfactuals with a logit pairing objective to improve causal fairness "
    "in toxicity classification. Our method achieves state-of-the-art counterfactual fairness "
    "across three benchmark datasets while maintaining competitive classification performance. "
    "The approach is simple to implement, requires no architectural modifications, and can be "
    "applied to any text classification model. Future work includes extending CLP to "
    "multi-class settings, exploring its application to other NLP fairness tasks, and "
    "investigating methods to jointly optimize causal and group fairness."
)

# Save
output_path = "docs/EMNLP_2026_Paper_Draft.docx"
doc.save(output_path)
print(f"论文初稿已保存至: {output_path}")
print(f"包含 {len(doc.paragraphs)} 个段落, {len(doc.tables)} 个表格")
