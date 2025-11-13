import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy

import seaborn; seaborn.set_style('whitegrid')
from apricot import FacilityLocationSelection

rng = numpy.random.RandomState(0)
X = numpy.concatenate([rng.normal((1, 1), 0.5, size=(15, 2)),
                       rng.normal((6, 3), 0.5, size=(25, 2)),
                       rng.normal((5, 7), 0.5, size=(40, 2)),
                       rng.normal((1, 7), 0.5, size=(30, 2)),
                       rng.normal((10, 4), 0.5, size=(15, 2)),
                       rng.normal((3, 4), 0.5, size=(15, 2))])

Xi = FacilityLocationSelection(6, 'euclidean').fit_transform(X)
Xr = rng.choice(numpy.arange(X.shape[0]), size=6)
Xr = X[Xr]

fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(111)
ax.scatter(X[:,0], X[:,1], s=10)
ax.legend(fontsize=14)
ax.set_xlim(-1, 14)
#ax.set_ylim(0, 7)
ax.axis('off')
plt.grid(False)
seaborn.despine(ax=ax)
fig.set_tight_layout(True)

def update(i):
	ax.clear()
	ax.scatter(X[:,0], X[:,1], s=10)
	ax.scatter(Xi[:i,0], Xi[:i,1], color="#FF6600", label="Submodular Selection")
	ax.scatter(Xr[:i,0], Xr[:i,1], color="#8A2BE2", label="Random Selection", alpha=0.6)
	ax.legend(fontsize=14)
	ax.set_xlim(-1, 14)
	ax.axis('off')
	plt.grid(False)
	return ax

anim = FuncAnimation(fig, update, frames=range(7), interval=1000)
anim.save('fl.gif', dpi=80, writer='imagemagick')