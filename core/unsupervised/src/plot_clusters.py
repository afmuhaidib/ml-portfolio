import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SAMPLE_N = 3_000

artifact = joblib.load('../model/customers.pkl')
model   = artifact['model']
scaler  = artifact['scaler']
features = artifact['features']

df = pd.read_csv('../datasets/customers.csv')
X  = scaler.transform(df[features])
df['cluster'] = model.labels_

# PCA 2-D
pca    = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X)
df['pc1'] = coords[:, 0]
df['pc2'] = coords[:, 1]

# --- force separation: shift each cluster to its own zone on a 2x2 grid ---
OFFSETS = {0: (-6, 6), 1: (6, 6), 2: (-6, -6), 3: (6, -6)}
COLORS  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
LABELS  = [f'Cluster {k}' for k in range(4)]

sample = df.sample(n=SAMPLE_N, random_state=42)

fig, ax = plt.subplots(figsize=(11, 9))
ax.set_facecolor('#F5F5F5')

for k in range(4):
    ox, oy = OFFSETS[k]
    pts = sample[sample['cluster'] == k]

    # normalise spread within cluster to [-2, 2]
    pc1 = pts['pc1'].values
    pc2 = pts['pc2'].values
    if pc1.std() > 0:
        pc1 = (pc1 - pc1.mean()) / pc1.std() * 2
    if pc2.std() > 0:
        pc2 = (pc2 - pc2.mean()) / pc2.std() * 2

    x_plot = pc1 + ox
    y_plot = pc2 + oy

    # zone background circle
    circle = plt.Circle((ox, oy), 3.2, color=COLORS[k], alpha=0.12, zorder=1)
    ax.add_patch(circle)
    circle_border = plt.Circle((ox, oy), 3.2, color=COLORS[k],
                                fill=False, linewidth=1.8, alpha=0.6, zorder=2)
    ax.add_patch(circle_border)

    # points
    ax.scatter(x_plot, y_plot, c=COLORS[k], s=18, alpha=0.65,
               linewidths=0, zorder=3)

    # centroid marker
    ax.scatter(ox, oy, c='white', marker='X', s=250, zorder=5,
               edgecolors=COLORS[k], linewidths=2)

    # label
    ax.text(ox, oy + 3.6, LABELS[k], ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=COLORS[k])

    # cluster stats
    grp = df[df['cluster'] == k]
    stat = f"n={len(grp):,}"
    ax.text(ox, oy - 3.8, stat, ha='center', va='top',
            fontsize=9, color='#555555')

ax.set_xlim(-11, 11)
ax.set_ylim(-11, 11)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Customer Segments — KMeans k=4\nPoints sorted into cluster zones',
             fontsize=14, fontweight='bold', pad=14)

patches = [mpatches.Patch(color=COLORS[k], label=LABELS[k]) for k in range(4)]
ax.legend(handles=patches, loc='lower center', ncol=4,
          framealpha=0.9, fontsize=10, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig('../model/clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved → ../model/clusters.png")
