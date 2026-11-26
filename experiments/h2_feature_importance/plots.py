"""Article figures from validated contributions; no external SHAP dependency."""
import argparse
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pcs5787-h2-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from core import FAMILIES, FAMILY_LABELS, seeded_int
from run_analysis import HERE, CONFIG, FEATURES, load_base, unit

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 200})
LABELS = {**FAMILY_LABELS, "all_meteorology": "Toda a meteorologia"}


def save(fig, directory, name):
    directory.mkdir(exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(directory / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def heatmap(table, title, label, out, name, vmin=None, vmax=None, cmap="viridis", annotate=False):
    fig, ax = plt.subplots(figsize=(max(6, .85*len(table.columns)), max(4, .33*len(table))))
    im = ax.imshow(table.to_numpy(), aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_title(title)
    if annotate:
        for i in range(len(table)):
            for j in range(len(table.columns)):
                ax.text(j, i, f"{table.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if table.iloc[i,j] < .65 else "black")
    fig.colorbar(im, ax=ax, label=label, shrink=.8)
    save(fig, out, name)


def beeswarm_jitter(x, rng):
    """Deterministic density stacking within x bins (equivalent summary display)."""
    width = np.ptp(x)
    bins = np.zeros(len(x), dtype=int) if width == 0 else np.minimum(69, ((x-x.min()) / width * 70).astype(int))
    jitter = np.zeros(len(x))
    for b in np.unique(bins):
        ix = np.flatnonzero(bins == b)
        rng.shuffle(ix)
        order = np.arange(len(ix))
        offsets = ((order+1)//2) * np.where(order % 2, 1, -1)
        jitter[ix] = offsets
    if np.max(np.abs(jitter)):
        jitter *= .38 / np.max(np.abs(jitter))
    return jitter


def make_shap(split):
    directory = HERE / split
    out = directory / "figures"
    variables = pd.read_csv(directory / "shap_variables.csv")
    glob = variables.loc[variables.scope.eq("global")].sort_values("mean_abs_joint", ascending=False)
    families = pd.read_csv(directory / "shap_families.csv")
    fam = families.loc[families.scope.eq("global") & families.family.isin(FAMILIES)].set_index("family").loc[list(FAMILIES)]
    subtitle = "avaliação retrospectiva, abr–set/2026" if split == "evaluation" else "desenvolvimento F1–F3, binários V3 equivalentes nas previsões"
    for n in [20, 51]:
        top = glob.head(n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, max(6, .26*n)))
        y = np.arange(n)
        ax.barh(y-.14, top.mean_abs_value, height=.25, label="Valor: média |SHAP|", color="#28608e")
        ax.barh(y+.14, top.mean_abs_flag, height=.25, label="Flag de ausência: média |SHAP|", color="#ce8a33")
        ax.scatter(top.mean_abs_joint, y, s=22, color="black", label="Conjunto: média |SHAP valor + flag|", zorder=3)
        ax.set_yticks(y, top.feature)
        ax.set_xlabel("Importância na margem bruta (log-odds)")
        ax.set_title(f"Atributos meteorológicos: {subtitle}")
        ax.legend(fontsize=8, loc="lower right")
        save(fig, out, f"shap_global_top{n}")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = np.arange(4)
    ax.barh(y-.18, fam.mean_abs_joint, height=.32, label="Média |soma das contribuições da família|", color="#28608e")
    ax.barh(y+.18, fam.sum_logical_mean_abs, height=.32, label="Soma das médias |contribuição de cada variável|", color="#58a892")
    ax.set_yticks(y, [LABELS[f] for f in fam.index])
    ax.set_xlabel("Log-odds; medidas distintas, sem normalização em porcentagens")
    ax.set_title(f"Importância por família: {subtitle}")
    ax.legend(fontsize=8)
    save(fig, out, "shap_families")

    obs = pd.read_parquet(directory / "observations.parquet")
    phi = np.load(directory / "shap_contributions.npz")["phi"]
    base = load_base().set_index(["time_local", "region"])
    values = base.loc[pd.MultiIndex.from_frame(obs[["time_local", "region"]]), FEATURES].to_numpy()
    joint = phi[:, 6:57] + phi[:, 57:108]
    top_names = glob.head(20).feature.tolist()[::-1]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    for row, name in enumerate(top_names):
        j = FEATURES.index(name)
        rng = np.random.default_rng(seeded_int(CONFIG["visual_seed"], split, name))
        sample = np.sort(rng.choice(len(obs), size=min(4000, len(obs)), replace=False))
        x = joint[sample, j]
        jitter = beeswarm_jitter(x, rng)
        raw = values[:, j]
        valid = np.isfinite(raw)
        color = np.full(len(raw), np.nan)
        color[valid] = rankdata(raw[valid]) / valid.sum()
        ok = valid[sample]
        ax.scatter(x[ok], row+jitter[ok], c=color[sample][ok], cmap="coolwarm", vmin=0, vmax=1,
                   s=4, alpha=.5, linewidths=0, rasterized=True)
        ax.scatter(x[~ok], row+jitter[~ok], c="#747474", marker="x", s=8, alpha=.55, linewidths=.5,
                   rasterized=True)
    ax.axvline(0, color="black", linewidth=.7)
    ax.set_yticks(range(20), [f"{name} [{unit(name)}]" for name in top_names], fontsize=8)
    ax.set_xlabel("SHAP do valor + flag de ausência (log-odds)")
    ax.set_title(f"Beeswarm: {subtitle}\nCinza × = ausente; até 4000 linhas/atributo, somente para exibição", fontsize=11)
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="coolwarm"), ax=ax,
                 label="Percentil do valor meteorológico válido", shrink=.65)
    save(fig, out, "shap_beeswarm")
    for scope in ["origin", "region"]:
        table = variables.loc[variables.scope.eq(scope)].pivot(index="feature", columns="group", values="mean_abs_joint")
        table = table.reindex(glob.head(20).feature)
        heatmap(table, f"SHAP por {'origem' if scope == 'origin' else 'região'}: {subtitle}",
                "Média |SHAP valor + flag| (log-odds)", out, f"shap_{scope}")
    stab = pd.read_csv(directory / "shap_stability.csv")
    for scope in ["origin", "region"]:
        table = stab.loc[stab.scope.eq(scope)].pivot(index="left", columns="right", values="spearman")
        heatmap(table, f"Concordância dos rankings SHAP entre {scope}", "Spearman, 51 variáveis",
                out, f"shap_rank_stability_{scope}", vmin=-1, vmax=1, cmap="coolwarm", annotate=True)

    if split == "evaluation":
        selected = pd.read_csv(directory / "local_example_selection.csv")
        local = pd.read_csv(directory / "local_contributions.csv")
        for origin, examples in selected.groupby("origin"):
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            for ax, row in zip(axes.ravel(), examples.itertuples()):
                sub = local.loc[local.row_id.eq(row.row_id)].copy()
                sub["abs"] = sub.contribution.abs()
                sub = sub.sort_values("abs", ascending=False, kind="stable")
                top = sub.head(8)
                contributions = top.contribution.tolist() + [sub.iloc[8:].contribution.sum()]
                names = top.feature.tolist() + ["Demais contribuições"]
                current = float(sub.base_value.iloc[0])
                ax.scatter(current, -1, color="black", marker="D", zorder=4)
                for i, value in enumerate(contributions):
                    ax.barh(i, abs(value), left=min(current, current+value), height=.65,
                            color="#b7333d" if value >= 0 else "#28608e")
                    ax.plot([current, current], [i-.6, i-.35], color="#888888", linewidth=.6)
                    current += value
                assert abs(current - sub.raw_margin.iloc[0]) < CONFIG["margin_atol"]
                ax.scatter(current, len(contributions), color="black", marker="D", zorder=4)
                ax.set_yticks(range(-1, len(contributions)+1), ["Valor-base"]+names+["Margem final"], fontsize=8)
                ax.invert_yaxis()
                ax.axvline(0, color="#888888", linewidth=.6)
                ax.set_xlabel("Margem bruta (log-odds)")
                ax.set_title(f"{row.cell} | {row.region} | {row.time_local}\ny={int(row.y)}, p={row.p:.3f}, limiar={row.threshold:.2f}", fontsize=10)
            for ax in axes.ravel()[len(examples):]: ax.set_visible(False)
            fig.suptitle(f"Explicações locais {origin}: seleção pela mediana de p em cada célula\nVermelho aumenta a margem; azul reduz; valores e flags somados", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, .94], w_pad=4)
            save(fig, out, f"local_{origin}")


def pi_chart(frame, names, labels, title, out, name):
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4.5, .36*len(names))), sharey=True)
    for ax, metric, label in zip(axes, ["i_ap", "i_brier"], ["AP original − AP permutada", "Brier permutado − Brier original"]):
        for hours, offset, color in [(24, -.14, "#28608e"), (168, .14, "#b15f24")]:
            sub = frame.loc[frame.hours.eq(hours)].set_index("target").loc[names]
            y = np.arange(len(names))+offset
            ax.hlines(y, sub[f"{metric}_p05"], sub[f"{metric}_p95"], color=color, linewidth=1.6)
            ax.scatter(sub[f"{metric}_mean"], y, color=color, label=f"Blocos {hours} h", s=24, zorder=3)
        ax.axvline(0, color="#777777", linewidth=.7)
        ax.set_xlabel(label)
        ax.set_yticks(range(len(names)), labels, fontsize=9)
        ax.grid(axis="x", alpha=.15)
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=9)
    fig.suptitle(title+"\nPontos: médias; segmentos: quantis Monte Carlo 5–95%, 20 repetições", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .92])
    save(fig, out, name)


def make_pi(split):
    directory = HERE / split
    out = directory / "figures"
    df = pd.read_csv(directory / "permutation_summary.csv")
    glob = df.loc[df.scope.eq("global")]
    subtitle = "avaliação retrospectiva, abr–set/2026" if split == "evaluation" else "desenvolvimento F1–F3"
    ranking = glob.loc[glob.kind.eq("variable") & glob.hours.eq(24)].sort_values("i_ap_mean", ascending=False)
    names = ranking.head(15).target.tolist()
    pi_chart(glob, names, names, f"Dependência preditiva por variável: {subtitle}", out, "permutation_variables")
    names = list(FAMILIES) + ["all_meteorology"]
    pi_chart(glob, names, [LABELS[n] for n in names], f"Dependência preditiva por família: {subtitle}", out, "permutation_families")
    for hours in CONFIG["block_hours"]:
        for scope in ["origin", "region"]:
            subset = df.loc[df.scope.eq(scope) & df.kind.isin(["family", "all"]) & df.hours.eq(hours)]
            for metric in ["i_ap", "i_brier"]:
                table = subset.pivot(index="target", columns="group", values=f"{metric}_mean").reindex(names)
                table.index = [LABELS[n] for n in table.index]
                bound = max(abs(table.to_numpy().min()), abs(table.to_numpy().max()), .001)
                heatmap(table, f"Permutação {hours} h, {'origem' if scope == 'origin' else 'região'}, {subtitle}", metric+" (média de 20 repetições)",
                        out, f"permutation_{scope}_{hours}_{metric}", vmin=-bound, vmax=bound, cmap="coolwarm")
    ranks = pd.read_csv(directory / "ranking_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in zip(axes, ["i_ap_24", "i_brier_24"], ["I_AP, blocos 24 h", "I_Brier, blocos 24 h"]):
        for family, idx in FAMILIES.items():
            sub = ranks.loc[ranks.feature.isin([FEATURES[i] for i in idx])]
            ax.scatter(sub.mean_abs_joint, sub[metric], label=LABELS[family], s=30, alpha=.8)
        to_label = set(ranks.nlargest(3, metric).feature) | set(ranks.nlargest(3, "mean_abs_joint").feature)
        for row in ranks.loc[ranks.feature.isin(to_label)].itertuples():
            ax.annotate(row.feature, (row.mean_abs_joint, getattr(row, metric)), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="gray", linewidth=.6)
        ax.set_xlabel("Média |SHAP valor + flag| (log-odds)")
        ax.set_ylabel(label)
    axes[1].legend(fontsize=7)
    fig.suptitle(f"SHAP e permutação medem propriedades diferentes: {subtitle}")
    fig.tight_layout()
    save(fig, out, "shap_permutation_comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["shap", "permutation"])
    parser.add_argument("--split", choices=CONFIG["splits"], default="evaluation")
    args = parser.parse_args()
    if args.stage == "shap": make_shap(args.split)
    else: make_pi(args.split)
