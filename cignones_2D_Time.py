# Imports
from fipy import (
    CellVariable,
    FaceVariable,
    Grid2D,
    ExponentialConvectionTerm,
    TransientTerm,
    DiffusionTerm,
    ImplicitSourceTerm,
    Viewer,
    Matplotlib2DGridViewer,
)
from fipy.tools import numerix
from fipy.variables.variable import Variable
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

# Model parameters
D = Variable()
ux = Variable()
uy = Variable()
q = 0

validation = False


Smax = 3.4  # Max capacity = 1000 cigognes/Km^2
Lx, Ly = 5500, 5500  # Domaine [km]
Nx, Ny = 100, 100  # Nombre de points de grille
dx, dy = Lx / Nx, Ly / Ny  # Taille d'un point de la grille (55/55)
Cx, Cy = 1.35 * Lx / 4, 3.2 * Ly / 4  # Point de départ de la distribution --> France
if validation:
    Cx, Cy = (
        1.55 * Lx / 4,
        3.6 * Ly / 4,
    )  # Point de départ de la distribution --> France
sigma = Lx / 10  # Coefficient de régulation de la distribution de départ
T = 42  # Temps de simulation [jour]
if validation:
    T = 50
dt = T / 1000  # dt sur lequel est résolu l'équation différentielle
Np = 10  # Number of snapshots of the solution (one every Nt/Np time steps)
Nt = Np * np.ceil(T / (Np * dt)).astype("int")  # Nt must be an integer divisible by Np


# Define the grid/mesh
mesh = Grid2D(nx=Nx, ny=Ny, dx=dx, dy=dy)
x, y = mesh.cellCenters[0], mesh.cellCenters[1]

# Define the model variable and set the boundary conditions
phi = CellVariable(
    name="numerical solution",
    mesh=mesh,
    value=Smax * numerix.exp(-((x - Cx) ** 2 + (y - Cy) ** 2) / (0.05 * sigma ** 2)),
)
meshBnd = mesh.facesLeft | mesh.facesRight | mesh.facesBottom | mesh.facesTop


# Impose zero flux on all 4 boundaries, Neumann
phi.faceGrad.constrain(0.0, where=meshBnd)

# Define and then solve the equation
eq = TransientTerm() == DiffusionTerm(coeff=D) - ExponentialConvectionTerm(
    coeff=(
        ux * FaceVariable(mesh=mesh, value=[1, 0])
        + uy * FaceVariable(mesh=mesh, value=[0, 1])
    )
) + ImplicitSourceTerm(coeff=q)
my_sol = np.zeros((Np, Nx * Ny))  # Matrix with Np solution snapshots
my_sol[0, :] = phi
k = 1

# Resolution of the equation
for step in np.arange(1, Nt):
    if validation:
        if (
            step < 500
        ):  # Setting the switch of parameters in Gibraltar : manually adjusted
            D.setValue(0)
            ux.setValue(-30)
            uy.setValue(-84)

        else:
            D.setValue(2000)
            ux.setValue(34)
            uy.setValue(-82)
    else:
        if (
            step < 400
        ):  # Setting the switch of parameters in Gibraltar : manually adjusted
            D.setValue(0)
            ux.setValue(-30)
            uy.setValue(-84)

        else:
            D.setValue(2000)
            ux.setValue(34)
            uy.setValue(-82)

    eq.solve(var=phi, dt=dt)
    if np.mod(step, Nt / Np) == 0:
        print(step, k)
        my_sol[k, :] = phi
        k += 1


# Plot the solution
fig, ax = plt.subplots(1, 1)
img = plt.imread(
    os.path.dirname(os.path.realpath(__file__)) + "/map.png"
)  # Google map image used for visualization

ax.imshow(img, extent=[0, 5500, 0, 5500])


xg, yg = np.meshgrid(np.linspace(0, Lx, Nx + 1), np.linspace(0, Ly, Ny + 1))
xd, yd = np.meshgrid(np.linspace(dx / 2, Lx, Nx), np.linspace(dy / 2, Ly, Ny))
for i in np.arange(Np):
    sol = my_sol[i, :].reshape((Nx, Ny))
    sol = griddata((xd.ravel(), yd.ravel()), my_sol[i, :], (xg, yg), method="nearest")
    plt.contourf(xg, yg, sol, zorder=0)
    plt.clim(0, 1)
    ax.set_aspect("equal", "box")
    plt.imshow(img, zorder=1, extent=[0, 5500, 0, 5500], alpha=0.25)
    # plt.savefig(
    #    "./Plots/fipy_advdiff_" + str(i) + ".png", bbox_inches="tight", pad_inches=0.0
    # )
    plt.show()
