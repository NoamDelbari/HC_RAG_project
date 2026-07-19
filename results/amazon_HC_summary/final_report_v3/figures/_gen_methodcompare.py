import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Real algorithm-comparison numbers (category-and-price corpus, pool N=1000),
# ordered by F1 descending. No internal dataset name.
methods = ['HC\n+ recal', 'BJ\n+ recal', 'Top-50', 'Top-20', 'Adaptive-K', 'CAR', 'BH', 'Bonf.']
R  = [0.209, 0.238, 0.287, 0.179, 0.104, 0.598, 0.064, 0.014]
P  = [0.209, 0.176, 0.139, 0.192, 0.204, 0.060, 0.087, 0.049]
F1 = [0.209, 0.202, 0.187, 0.185, 0.138, 0.110, 0.074, 0.022]

x = np.arange(len(methods))
w = 0.27
fig, ax = plt.subplots(figsize=(7.6, 3.8))

# highlight the winning method
ax.axvspan(x[0] - 0.5, x[0] + 0.5, color='#fff3cd', zorder=0)

bR = ax.bar(x - w, R,  w, label='Recall',    color='#4c72b0')
bP = ax.bar(x,     P,  w, label='Precision', color='#dd8452')
bF = ax.bar(x + w, F1, w, label='F1',        color='#55a868')

ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, 0.64)
ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax.set_title('All eight cut-off methods (category-and-price corpus, pool $N{=}1000$):\n'
             'HC with recalibration has the best F1 and a balanced recall/precision',
             fontsize=10.5)
ax.grid(axis='y', alpha=0.25)

# label the F1 of every method (the headline metric)
for r, v in zip(bF, F1):
    ax.annotate('%.3f' % v, (r.get_x() + r.get_width() / 2, v + 0.006),
                ha='center', va='bottom', fontsize=7.5, color='#2f6f47')

# call-outs
ax.annotate('best F1', xy=(x[0] + w, 0.209), xytext=(x[0] + 0.15, 0.40),
            ha='center', fontsize=9, color='#2f6f47', fontweight='bold',
            arrowprops=dict(arrowstyle='-|>', color='#2f6f47', lw=1.3))
ax.text(x[5], 0.61, 'over-retrieves\n(R 0.60, P 0.06)', ha='center', va='top',
        fontsize=8, style='italic', color='#555555')
ax.text((x[6] + x[7]) / 2, 0.16, 'collapse', ha='center', fontsize=8,
        style='italic', color='#555555')

plt.tight_layout()
plt.savefig('fig_methodcompare.png', dpi=150, bbox_inches='tight')
print('saved fig_methodcompare.png')
