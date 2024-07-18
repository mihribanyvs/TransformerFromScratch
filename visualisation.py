
#Visualizations
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 8))

def update(frame):
    ax.cla()  # Clear the current axes
    for i in range(frame):
        phi_psi = angles_original[i]
        ax.scatter(phi_psi[0, :], phi_psi[1, :], s=10, alpha=0.5, color='#2E3532')
    phi_psi = angles_original[frame]
    ax.scatter(phi_psi[0, :], phi_psi[1, :], s=10, alpha=0.5, color='#8B2635')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel(r'$\phi$ (degrees)')
    ax.set_ylabel(r'$\psi$ (degrees)')
    ax.set_title(f'Ramachandran Plot for Protein {frame+1}')
    ax.grid(True)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.5)


ani = FuncAnimation(fig, update, frames=range(angles_original.shape[0]), repeat=True)

ani.save('ramachandran_plots_test.mp4', writer='ffmpeg', fps=2)

plt.show()

psi_original = angles_original[:, 1, :]
psi_pred = angles_pred[:, 1, :]
psi_pred = psi_pred[psi_pred != 0]
psi_original = psi_original[psi_original != 0]

plt.figure(figsize=(10, 6))
sns.kdeplot(psi_original, label='Original ψ Angles', color='#2E3532', fill=True, alpha=0.5)
sns.kdeplot(psi_pred, label='Predicted ψ Angles', color='#8B2635', fill=True, alpha=0.5)
plt.xlabel('ψ Angles (degrees)')
plt.ylabel('Density')
plt.title(f'Distribution of ψ Angles for 50 proteins')
plt.legend()
plt.grid(True)
plt.show()