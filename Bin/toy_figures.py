import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

np.random.seed(0)
n = 120
t = np.linspace(0, 10, n)                          
x = np.cumsum(0.6*np.random.randn(n)) + 10*np.sin(t/2)
y = np.cumsum(0.6*np.random.randn(n)) + 10*np.cos(t/2)
prob = np.clip(0.15 + 0.7*(np.sin(1.1*t) + 0.25*np.random.randn(n))/2 + 0.4, 0, 1)
y_true = (np.sin(1.1*t) > 0.65).astype(int)         

# Figure layout 
fig = plt.figure(figsize=(12, 7))
gs = GridSpec(
    2, 3, figure=fig,
    width_ratios=[1.0, 1.0, 1.2],   
    height_ratios=[1.0, 0.65]      
)
plt.subplots_adjust(wspace=0.35, hspace=0.35)

# Panel A: 2D spatial trajectory (color=time, size=prob)
axA = fig.add_subplot(gs[0, 0])
scA = axA.scatter(x, y, c=t, s=prob*250+20, cmap='plasma',
                  edgecolor='k', linewidth=0.4, alpha=0.9)
axA.plot(x, y, color='gray', lw=1, alpha=0.5, zorder=0)
axA.set_title('A  Spatial trajectory (color = time, size = prob)', loc='left')
axA.set_xlabel('X (μm)'); axA.set_ylabel('Y (μm)')
axA.axis('equal')
cbarA = plt.colorbar(scA, ax=axA, fraction=0.046, pad=0.04)
cbarA.set_label('Time (hours)')

# Panel B: 2D spatial map (color=prob)
axB = fig.add_subplot(gs[0, 1])
scB = axB.scatter(x, y, c=prob, s=60, cmap='viridis',
                  edgecolor='k', linewidth=0.4, alpha=0.95)
axB.plot(x, y, color='lightgray', lw=1, alpha=0.6, zorder=0)
axB.set_title('B  Spatial probability map (color = predicted prob)', loc='left')
axB.set_xlabel('X (μm)'); axB.set_ylabel('Y (μm)')
axB.axis('equal')
cbarB = plt.colorbar(scB, ax=axB, fraction=0.046, pad=0.04)
cbarB.set_label('Predicted event probability')

# Panel C: 3D x–y–t trajectory
axC = fig.add_subplot(gs[0, 2], projection='3d')
axC.plot(x, y, t, color='steelblue', lw=2, alpha=0.9)
axC.scatter(x, y, t, c=prob, cmap='plasma', s=14, alpha=0.9, edgecolor='k')
axC.set_title('C  3D spatiotemporal trajectory', loc='left', pad=12)
axC.set_xlabel('X (μm)'); axC.set_ylabel('Y (μm)'); axC.set_zlabel('Time (h)')

# Panel D: wide probability-over-time with ground truth
axD = fig.add_subplot(gs[1, :])
axD.plot(t, prob, lw=2.2, label='Predicted probability', color='tab:blue')
axD.plot(t, y_true, 'k--', lw=1.2, label='True event (binary)')
thr = 0.5
axD.axhline(thr, color='gray', lw=1, ls=':', alpha=0.8)
axD.text(t[3], thr+0.02, 'threshold', color='gray')
axD.set_title('D  Probability over time', loc='left')
axD.set_xlabel('Time (hours)'); axD.set_ylabel('Probability / Event')
axD.set_ylim(-0.05, 1.05)
axD.legend(frameon=False)

plt.tight_layout()
plt.show()
