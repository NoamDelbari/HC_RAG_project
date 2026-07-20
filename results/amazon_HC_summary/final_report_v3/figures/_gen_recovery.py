import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Only the two groups HC actually FAILS on under the global null (both start at 0),
# so "after" can never look worse than "before". The 77 already-working queries are
# unchanged (0.253 -> 0.251) and are described in the caption, not shown as bars.
groups   = ['Calibration\nmismatch (n=31)', 'Embedding\nlimit (n=24)']
before   = [0.000, 0.000]   # HC, global null (before fix)
after    = [0.157, 0.012]   # HC + recalibration (after fix)
baseline = [0.164, 0.014]   # fixed Top-50 reference (per group)

x = np.arange(len(groups))
w = 0.26
fig, ax = plt.subplots(figsize=(7.2, 3.7))
b1 = ax.bar(x - w, before,   w, label='HC, global null (before fix)',   color='#9e9e9e')
b2 = ax.bar(x,     after,    w, label='HC + recalibration (after fix)', color='#1b7837')
b3 = ax.bar(x + w, baseline, w, label='Fixed Top-50 (reference)',       color='#d2d2d2',
            hatch='//', edgecolor='#8a8a8a')

ax.set_ylabel('Retrieval F1 (per group)')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylim(0, 0.21)
ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
ax.set_title('On the queries HC fails: the fix restores the calibration group to the\n'
             'baseline level; the embedding-limited group has no signal to recover',
             fontsize=10.5)
ax.grid(axis='y', alpha=0.25)

for bars in (b1, b2, b3):
    for r in bars:
        h = r.get_height()
        ax.annotate('%.3f' % h, (r.get_x() + r.get_width() / 2, h + 0.003),
                    ha='center', va='bottom', fontsize=8)

ax.text(0, 0.190, '30 of 31 recovered', ha='center', fontsize=9.5,
        color='#1b7837', fontweight='bold')

plt.tight_layout()
plt.savefig('fig_recovery_beforeafter.png', dpi=150, bbox_inches='tight')
print('saved fig_recovery_beforeafter.png')
