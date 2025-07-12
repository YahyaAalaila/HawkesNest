import io
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde
import matplotlib.patheffects as PathEffects
import pandas as pd
import imageio


# remove any normalization logic
def plot_kde(
    coords: np.ndarray,
    savepath: str,
    dataset_name: str,
    domain: list,
    text: str | None = None,
    name: str | None = None,
):
    """
    coords: array of shape (n_samples, 2) with raw x/y or lon/lat
    savepath: directory into which to save
    dataset_name: used for map lookup or naming
    text: optional annotation string
    name: optional override of output filename (defaults to "{dataset_name}_density")
    """
    if name is None:
        name = f"{dataset_name}_density"

    # just extract columns
    xs = coords[:, 0]
    ys = coords[:, 1]

    _, ax = plt.subplots(figsize=(6, 6))

    # build an axis-aligned KDE
    kernel = gaussian_kde(np.vstack([xs, ys]))

    # grid over the bounding box (or auto–min/max of data if you prefer)
    xmin, xmax, ymin, ymax = domain
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kernel(positions).reshape(X.shape)

    ax.contourf(X, Y, Z, levels=10, alpha=0.6, cmap="RdGy")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if text is not None:
        txt = ax.text(
            0.15, 0.9, text,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            size=16,
            color="white"
        )
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground="black")])

    plt.axis("off")

    out_dir = os.path.join(savepath, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{name}.png"), bbox_inches="tight", dpi=150)
    plt.close()
    
def plot_figure_7_2_style(h_vals, t_vals, L_diff, savepath, costum_name):
    """
    Create a contour plot similar to Figure 7.2 in the manuscript,
    with x-axis = temporal bandwidth, y-axis = spatial bandwidth,
    and contour lines = L_diff values.
    """
    
    # Create meshgrid so that X is the temporal dimension, Y is the spatial dimension
    X, Y = np.meshgrid(t_vals, h_vals)  # shape => (len(h_vals), len(t_vals))

    plt.figure(figsize=(12,8))
    
    # Choose contour levels that make sense for your data range
    # For example, if L_diff ranges from 0 to ~450, pick levels accordingly
    # You can set them manually or use np.linspace(...) to generate them.
    min_val = np.nanmin(L_diff)
    max_val = np.nanmax(L_diff)
    num_levels = 10  # or however many contour lines you want

    # Create evenly spaced levels from min to max
    levels = np.linspace(min_val, max_val, num_levels)
    
    # Create contour lines (no color fill)
    CS = plt.contour(X, Y, L_diff, levels=levels, colors='black',  linewidths=2.5)
    
    # Label the contour lines
    plt.clabel(CS, inline=True, fmt='%1.1f', fontsize=20)
    
    # Axes labels and title
    matplotlib.rcParams.update({'font.size': 20})  # Set global font size
    plt.xlabel("Temporal bandwidth (days)")
    plt.xlabel("Temporal bandwidth (days)")
    plt.ylabel("Spatial bandwidth (m)")
    
    # plt.title("Global K function: observed - upper envelope")
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=20)  
    ax.set_xlabel("Temporal bandwidth (days)", fontsize=20)
    ax.set_ylabel("Spatial bandwidth (m)", fontsize=20)
    for spine in ax.spines.values():
        spine.set_linewidth(2)           # ← thicker frame
        out_dir = os.path.join(savepath, costum_name)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{costum_name}.png"), bbox_inches="tight", dpi=300)
    plt.close()
    
    

import os
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import patheffects


# def plot_kde_gif(
#     df: pd.DataFrame,
#     savepath: str,
#     dataset_name: str,
#     domain: list | None = None,
#     name: str | None = None,
#     n_frames: int = 60,
#     fps: int = 2,
#     cmap: str = "RdGy",
#     grid_size: int = 100,
#     dpi: int = 150,
#     # mark overlay configuration:
#     mark_strategy: str = "sample",  # one of ['all', 'sample', 'topk']
#     mark_n: int = 20,                # number of marks to show when sampling or topk
#     mark_lifetime_sec: float = 1.0
# ):
#     """
#     Animate the KDE of (x,y) points over time, saving a GIF.

#     You can overlay a subset of points (×) that reflect the density:
#       - 'all': show all original events (default behavior)
#       - 'sample': sample `mark_n` new points from the KDE (most probable locations)
#       - 'topk': pick `mark_n` original events with highest density at their locations

#     Marks appear for `mark_lifetime_sec` playback seconds, then fade.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Must contain columns ["t", "x", "y"].
#     savepath : str
#         Directory into which to save the GIF.
#     dataset_name : str
#         Used for naming the subfolder and file.
#     domain : list [xmin, xmax, ymin, ymax], optional
#         Bounding box for the grid; auto-computed if None.
#     name : str, optional
#         Override output filename (defaults to "{dataset_name}_evolution.gif").
#     n_frames : int
#         Number of time-slices / frames.
#     fps : int
#         Frames per second in the final GIF.
#     cmap : str
#         Matplotlib colormap for the filled contours.
#     grid_size : int
#         Number of grid points along each axis when evaluating the KDE.
#     dpi : int
#         Resolution for each frame PNG.
#     mark_strategy : str
#         Strategy for selecting overlay marks: 'all', 'sample', or 'topk'.
#     mark_n : int
#         Number of marks when sampling or topk.
#     mark_lifetime_sec : float
#         How many seconds of playback each mark stays visible.
#     """
#     # Prepare output paths
#     if name is None:
#         name = f"{dataset_name}_evolution.gif"
#     out_dir = os.path.join(savepath, dataset_name)
#     os.makedirs(out_dir, exist_ok=True)
#     out_gif = os.path.join(out_dir, name)

#     # Sort by time
#     df_sorted = df.sort_values("t").reset_index(drop=True)
#     xs = df_sorted["x"].to_numpy()
#     ys = df_sorted["y"].to_numpy()
#     ts = df_sorted["t"].to_numpy()

#     # Domain
#     if domain is None:
#         xmin, xmax = xs.min(), xs.max()
#         ymin, ymax = ys.min(), ys.max()
#     else:
#         xmin, xmax, ymin, ymax = domain

#     # Time grid
#     t_min, t_max = ts.min(), ts.max()
#     times = np.linspace(t_min, t_max, n_frames)
#     dt_sim = times[1] - times[0] if n_frames > 1 else (t_max - t_min)
#     lifetime_sim = dt_sim * (mark_lifetime_sec * fps)

#     # Precompute KDE evaluation grid
#     X, Y = np.mgrid[xmin:xmax:grid_size*1j, ymin:ymax:grid_size*1j]
#     positions = np.vstack([X.ravel(), Y.ravel()])

#     frames = []
#     for t_cut in times:
#         past = df_sorted[df_sorted["t"] <= t_cut]
#         pts = past[["x", "y"]].to_numpy()
#         # KDE
#         if pts.shape[0] > pts.shape[1]:
#             kernel = gaussian_kde(pts.T)
#             Z = kernel(positions).reshape(X.shape)
#         else:
#             Z = np.zeros_like(X)

#         # Prepare figure
#         fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
#         ax.contourf(X, Y, Z, levels=10, alpha=0.6, cmap=cmap)
#         ax.set_xlim(xmin, xmax)
#         ax.set_ylim(ymin, ymax)
#         ax.axis("off")

#         # Overlay marks selection window
#         window_mask = (df_sorted["t"] <= t_cut) & (df_sorted["t"] > (t_cut - lifetime_sim))
#         data_window = df_sorted[window_mask]

#         if mark_strategy == 'all':
#             mark_pts = data_window[['x','y']].to_numpy()
#         elif mark_strategy == 'sample' and pts.shape[0] > pts.shape[1]:
#             # sample new points from KDE
#             sampled = kernel.resample(mark_n).T
#             mark_pts = sampled
#         elif mark_strategy == 'topk' and pts.shape[0] >= mark_n and pts.shape[0] > pts.shape[1]:
#             # evaluate density at original past points
#             dens = kernel(pts.T)
#             idx = np.argsort(dens)[-mark_n:]
#             mark_pts = pts[idx]
#         else:
#             mark_pts = np.empty((0,2))

#         # Plot marks
#         if mark_pts.size:
#             ax.scatter(
#                 mark_pts[:,0], mark_pts[:,1],
#                 marker='x', s=50,
#                 color='black', linewidths=1.5,
#                 alpha=1.0
#             )

#         # Capture via in-memory PNG
#         buf = io.BytesIO()
#         fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
#         plt.close(fig)
#         buf.seek(0)
#         img = imageio.v2.imread(buf)
#         frames.append(img)

#     # Save GIF
#     imageio.mimsave(out_gif, frames, fps=fps)
#     print(f"Saved animated KDE with '{mark_strategy}' marks to {out_gif}")

def plot_kde_gif(
    df: pd.DataFrame,
    savepath: str,
    dataset_name: str,
    domain: list | None = None,
    name: str | None = None,
    n_frames: int = 60,
    fps: int = 2,
    cmap: str = "RdGy",
    grid_size: int = 100,
    dpi: int = 150,
    # mark overlay configuration:
    mark_col: str = "m",
    mark_strategy: str = "topk",  # one of ['all', 'sample', 'topk']
    mark_n: int = 10,               # number of marks to show
    mark_lifetime_sec: float = 1.0
):
    """
    Animate the KDE of (x,y) points over time, saving a GIF.

    Overlay a subset of marks (×) reflecting density and colored by `mark_col`:
      - 'all': show all events in window, colored by type
      - 'topk': pick `mark_n` original events with highest density, colored by type
      - 'sample': draw `mark_n` sample points from KDE (no types/colors)

    Marks appear for `mark_lifetime_sec` playback seconds, then disappear.
    """
    # Setup output
    if name is None:
        name = f"{dataset_name}_evolution.gif"
    out_dir = os.path.join(savepath, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_gif = os.path.join(out_dir, name)

    # Sort by time
    df_sorted = df.sort_values("t").reset_index(drop=True)
    xs = df_sorted["x"].to_numpy()
    ys = df_sorted["y"].to_numpy()
    ts = df_sorted["t"].to_numpy()

    # Domain
    if domain is None:
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
    else:
        xmin, xmax, ymin, ymax = domain

    # Time grid and lifetime in sim-time
    t_min, t_max = ts.min(), ts.max()
    times = np.linspace(t_min, t_max, n_frames)
    dt_sim = times[1] - times[0] if n_frames > 1 else (t_max - t_min)
    lifetime_sim = dt_sim * (mark_lifetime_sec * fps)

    # Precompute grid
    X, Y = np.mgrid[xmin:xmax:grid_size*1j, ymin:ymax:grid_size*1j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    frames = []
    for t_cut in times:
        # past events up to t_cut
        past = df_sorted[df_sorted["t"] <= t_cut]
        pts = past[["x", "y"]].to_numpy()
        # KDE
        if pts.shape[0] > pts.shape[1]:
            kernel = gaussian_kde(pts.T)
            Z = kernel(positions).reshape(X.shape)
        else:
            Z = np.zeros_like(X)

        # draw base KDE
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax.contourf(X, Y, Z, levels=10, alpha=0.6, cmap=cmap)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis("off")

        # determine which marks to overlay, within lifetime window
        window_df = past[past["t"] > (t_cut - lifetime_sim)]

        if mark_strategy == 'all':
            mark_df = window_df
        elif mark_strategy == 'topk' and window_df.shape[0] > window_df.shape[1]:
            # density at each window point
            win_pts = window_df[["x","y"]].to_numpy()
            dens_win = kernel(win_pts.T)
            idx_win = np.argsort(dens_win)[-mark_n:]
            mark_df = window_df.iloc[idx_win]
        elif mark_strategy == 'sample' and pts.shape[0] > pts.shape[1]:
            sampled = kernel.resample(mark_n).T
            mark_df = pd.DataFrame(sampled, columns=["x","y"])
            mark_df[mark_col] = None
        else:
            mark_df = pd.DataFrame(columns=["x","y", mark_col])

        # plot marks by type
        if not mark_df.empty:
            if mark_col in mark_df and mark_strategy != 'sample':
                cats = mark_df[mark_col].unique()
                cmap_pts = plt.get_cmap('tab20', len(cats))
                for i, cat in enumerate(cats):
                    sel = mark_df[mark_df[mark_col] == cat]
                    ax.scatter(sel['x'], sel['y'], marker='x', s=50,
                               color=cmap_pts(i), linewidths=1.5)
            else:
                ax.scatter(mark_df['x'], mark_df['y'], marker='x', s=50,
                           color='black', linewidths=1.5)

        # capture frame
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = imageio.v2.imread(buf)
        frames.append(img)

    # save
    imageio.mimsave(out_gif, frames, fps=fps)
    print(f"Saved animated KDE with '{mark_strategy}' marks to {out_gif}")


if __name__ == "__main__":
    import pandas as pd
    OUT_DIR = Path("quick_results")
    OUT_DIR.mkdir(exist_ok=True)
    CSV_OUT  = OUT_DIR / "events_ent_cluster_KDE.csv"
    
    
    # assume events_df has columns ["x","y","t", ...] 
    df = pd.read_csv("quick_results/data/events_topology_sine + separable 4_df.csv")  # or however you loaded it
    coords = df[["x","y"]].to_numpy()
    x_min, x_max = coords[:,0].min(), coords[:,0].max()
    y_min, y_max = coords[:,1].min(), coords[:,1].max()
    # choose bounds: if your simulation is in [0,1]², then:
    bounds = (x_min, x_max, y_min, y_max)
    


    plot_kde_gif(
        df,
        savepath=OUT_DIR,
        dataset_name="topo_evolution",
        domain=bounds,
        n_frames=60,
        fps=4
    )

    #plot_kde(coords, dataset_name="ent",savepath="kde_plots", domain=bounds, name="poly-bg-kde")

