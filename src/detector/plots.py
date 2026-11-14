"""Figures for the final report and slides. One visual style for all of them.

Colour roles: human = blue, ChatGPT = orange (everywhere); model-comparison bars use one
neutral hue (violet) so they are never confused with the two classes.
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from sklearn.metrics import roc_curve    # noqa: E402

from .config import CHATGPT_COLOR, DOMAIN_NAMES, HUMAN_COLOR   # noqa: E402

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
MODEL_COLOR = '#4a3aa7'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


def setup_style():
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 200, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'font.family': ['Segoe UI', 'DejaVu Sans', 'sans-serif'],
        'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.titlecolor': INK, 'axes.labelcolor': INK_2, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False, 'xtick.color': MUTED, 'ytick.color': INK_2,
        'xtick.labelcolor': INK_2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.axisbelow': True, 'legend.frameon': False, 'legend.fontsize': 9,
    })


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def metric_bars(table, metric, title, path, xlabel=None, ci=None, xlim=(0, 1.0)):
    """Horizontal bars, one per model, sorted; optional 95% CI whiskers."""
    order = sorted(table, key=lambda m: table[m][metric])
    values = [table[m][metric] for m in order]
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(order) + 1.2))
    bars = ax.barh(order, values, color=MODEL_COLOR, height=0.62)
    if ci:
        lows = [max(values[i] - ci[m][0], 0) for i, m in enumerate(order)]
        highs = [max(ci[m][1] - values[i], 0) for i, m in enumerate(order)]
        ax.errorbar(values, order, xerr=[lows, highs], fmt='none', ecolor=INK_2, elinewidth=1, capsize=3)
    for bar, v, m in zip(bars, values, order):
        x = (ci[m][1] if ci else v) + 0.006          # the label starts after the confidence interval
        ax.text(min(x, xlim[1]), bar.get_y() + bar.get_height() / 2, f'{v:.3f}', va='center', fontsize=9, color=INK)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel or metric)
    ax.grid(axis='y', visible=False)
    ax.set_title(title)
    _save(fig, path)


def confusion_grid(results, names, path):
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.2))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        cm = np.array(results[name]['confusion_matrix'])
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b', [SURFACE, BLUES[5]]),
                  vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]:,}\n{share[i, j]:.1%}', ha='center', va='center', fontsize=9.5,
                        color='white' if share[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ['Human', 'ChatGPT'])
        ax.set_yticks([0, 1], ['Human', 'ChatGPT'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.grid(False)
        ax.set_title(name, fontsize=10)
    fig.suptitle('Confusion matrices on the test set (rows = true class)', x=0.01, ha='left',
                 fontsize=12, fontweight='bold', color=INK)
    _save(fig, path)


def roc_curves(curves, path):
    """curves: {name: (y_true, score)}; a zoomed panel shows the low false-positive region."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for k, (name, (y, s)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y, s)
        for ax in axes:
            ax.plot(fpr, tpr, color=SERIES[k % len(SERIES)], lw=2, label=name)
    axes[0].plot([0, 1], [0, 1], color=AXIS, lw=1, ls='--')
    axes[0].set_title('ROC curves (test set)')
    axes[1].set_xlim(0, 0.1)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].axvline(0.01, color=MUTED, lw=1, ls=':')
    axes[1].text(0.0105, 0.52, '1% of humans\nfalsely flagged', fontsize=8, color=INK_2)
    axes[1].set_title('Zoom: false-positive rate below 10%')
    for ax in axes:
        ax.set_xlabel('False positive rate (humans flagged as ChatGPT)')
        ax.set_ylabel('True positive rate (ChatGPT caught)')
    axes[1].legend(loc='lower right')
    _save(fig, path)


def class_histogram(values_human, values_chatgpt, path, title, xlabel, threshold=None, log=True, bins=40):
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    allv = np.concatenate([values_human, values_chatgpt])
    edges = (np.logspace(np.log10(max(allv.min(), 1e-3)), np.log10(allv.max()), bins) if log
             else np.linspace(allv.min(), allv.max(), bins))
    ax.hist(values_human, bins=edges, color=HUMAN_COLOR, alpha=0.65, label='Human')
    ax.hist(values_chatgpt, bins=edges, color=CHATGPT_COLOR, alpha=0.65, label='ChatGPT')
    if log:
        ax.set_xscale('log')
    if threshold is not None:
        ax.axvline(threshold, color=INK, lw=1.2, ls='--')
        ax.text(threshold, ax.get_ylim()[1] * 0.92, f'  threshold {threshold:.1f}', fontsize=8.5, color=INK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Texts')
    ax.legend()
    ax.set_title(title)
    _save(fig, path)


def lodo_heatmaps(lodo, models, path):
    """Left: F1 on the unseen domain. Right: drop compared with training on all domains."""
    domains = sorted({row['domain'] for row in lodo})
    get = lambda key: np.array([[next((r[key] for r in lodo if r['model'] == m and r['domain'] == d), np.nan)
                                 for d in domains] for m in models], dtype=float)
    unseen, drop = get('f1_unseen'), get('f1_drop')
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.5 * len(models) + 1.8))
    oranges = matplotlib.colors.LinearSegmentedColormap.from_list('o', [SURFACE, '#f6c3a8', CHATGPT_COLOR, '#a3401a'])
    blues = matplotlib.colors.LinearSegmentedColormap.from_list('b', [SURFACE] + BLUES[1:])
    show_f1 = lambda v: f'{v:.3f}'
    show_change = lambda v: '0.000' if abs(v) < 0.0005 else f'{-v:+.3f}'   # a drop of 0.131 is shown as "-0.131"
    panels = [(unseen, blues, 'F1 on the unseen domain', (0.5, 1.0), show_f1),
              (drop, oranges, 'Change of F1 vs. the domain seen in training',
               (0, max(0.15, float(np.nanmax(drop)))), show_change)]
    for ax, (values, cmap, title, (lo, hi), show) in zip(axes, panels):
        ax.imshow(values, cmap=cmap, vmin=lo, vmax=hi, aspect='auto')
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                if np.isnan(v):
                    ax.text(j, i, 'not tested', ha='center', va='center', fontsize=7.5, color=MUTED)
                    continue
                dark = (v - lo) / (hi - lo + 1e-9) > 0.6
                ax.text(j, i, show(v), ha='center', va='center', fontsize=8.5, color='white' if dark else INK)
        ax.set_xticks(range(len(domains)), [DOMAIN_NAMES.get(d, d) for d in domains], rotation=20, ha='right')
        ax.set_yticks(range(len(models)), models if ax is axes[0] else [''] * len(models))
        ax.grid(False)
        ax.set_title(title, fontsize=11)
    fig.suptitle('Generalisation: train on 4 domains, test on the 5th (same test texts for every model)',
                 x=0.01, ha='left', fontsize=12, fontweight='bold', color=INK)
    _save(fig, path)


def grouped_bars(categories, series, path, title, ylabel, colors, ylim=(0, 100), fmt='{:.1f}'):
    """series: {label: [values per category]}; the legend sits under the plot so it never covers bars."""
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    width = 0.8 / len(series)
    x = np.arange(len(categories))
    for k, (label, values) in enumerate(series.items()):
        bars = ax.bar(x + (k - (len(series) - 1) / 2) * width, values, width * 0.92, color=colors[k], label=label)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (ylim[1] - ylim[0]) * 0.012,
                    fmt.format(v), ha='center', va='bottom', fontsize=8, color=INK)
    ax.set_xticks(x, categories)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=min(len(series), 4))
    ax.set_title(title)
    _save(fig, path)


def diverging_bars(names, values, path, title, xlabel, left_label='← human', right_label='ChatGPT →'):
    order = np.argsort(values)
    names, values = np.array(names)[order], np.array(values)[order]
    fig, ax = plt.subplots(figsize=(7.6, 0.3 * len(names) + 1.3))
    ax.barh(names, values, color=[HUMAN_COLOR if v < 0 else CHATGPT_COLOR for v in values], height=0.66)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_xlabel(f'{xlabel}     ({left_label}  |  {right_label})')
    ax.grid(axis='y', visible=False)
    ax.set_title(title)
    _save(fig, path)


def importance_bars(names, values, errors, path, title, xlabel):
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(7.6, 0.3 * len(names) + 1.3))
    ax.barh(np.array(names)[order], np.array(values)[order], xerr=np.array(errors)[order], color=MODEL_COLOR,
            height=0.66, error_kw={'ecolor': INK_2, 'elinewidth': 0.8, 'capsize': 2})
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', visible=False)
    ax.set_title(title)
    _save(fig, path)


def training_curves(history, path, title, x_key='epoch', lines=(('train_loss', 'Training loss'),
                                                                     ('val_loss', 'Validation loss')),
                    best_x=None):
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    for k, (key, label) in enumerate(lines):
        xs = [h[x_key] for h in history if key in h]
        ys = [h[key] for h in history if key in h]
        ax.plot(xs, ys, color=SERIES[k], lw=2, marker='o' if len(xs) < 30 else None, ms=6, label=label)
    if best_x is not None:
        ax.axvline(best_x, color=MUTED, lw=1, ls='--')
        ax.text(best_x, ax.get_ylim()[1] * 0.95, '  best epoch (early stopping)', fontsize=8.5, color=INK_2,
                va='top')
    ax.set_xlabel(x_key.replace('_', ' '))
    ax.set_ylabel('Loss (binary cross-entropy)')
    ax.legend()
    ax.set_title(title)
    _save(fig, path)


def error_by_length(table, sizes, path):
    """table: {model: {bucket: {'false_positive_rate', 'miss_rate'}}}; sizes: {bucket: {'human', 'chatgpt'}}"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    panels = (('false_positive_rate', 'human', 'Humans wrongly flagged as ChatGPT'),
              ('miss_rate', 'chatgpt', 'ChatGPT texts missed'))
    for ax, (key, cls, title) in zip(axes, panels):
        for k, (model, rates) in enumerate(table.items()):
            labels = list(rates)
            ax.plot(range(len(labels)), [rates[b][key] * 100 for b in labels], color=SERIES[k % len(SERIES)],
                    lw=2, marker='o', ms=6, label=model)
        ax.set_xticks(range(len(labels)), [f'{b}\n(n={sizes[b][cls]})' for b in labels])
        ax.set_xlabel('Text length in words (number of texts of this class)')
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel('% of texts of this class')
    axes[1].legend(fontsize=8.5)
    fig.suptitle('Errors by text length on the test set: short ChatGPT answers are rare and often missed',
                 x=0.01, ha='left', fontsize=12, fontweight='bold', color=INK)
    _save(fig, path)
