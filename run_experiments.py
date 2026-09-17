import time
import numpy as np
from sklearn.datasets import fetch_covtype


def standard_kmeans(X, init_centroids, max_iters=8):
    n, d = X.shape
    k = len(init_centroids)
    centroids = init_centroids.copy()

    dist_computations = 0
    t0 = time.time()

    for it in range(max_iters):
        chunk_size = 5000
        labels = np.empty(n, dtype=np.int32)
        c_norm = np.sum(centroids**2, axis=1, keepdims=True).T

        for start in range(0, n, chunk_size):
            chunk = X[start:start + chunk_size]
            x_norm = np.sum(chunk**2, axis=1, keepdims=True)
            dists = x_norm + c_norm - 2.0 * np.dot(chunk, centroids.T)
            labels[start:start + chunk_size] = np.argmin(dists, axis=1)
            dist_computations += len(chunk) * k

        counts = np.bincount(labels, minlength=k)
        sums = np.zeros((k, d), dtype=np.float32)
        np.add.at(sums, labels, X)
        nonzero = counts > 0
        centroids[nonzero] = sums[nonzero] / counts[nonzero, None]

    runtime = time.time() - t0
    return centroids, runtime, dist_computations


def fast_mkm(X, init_centroids, m_groups=32, top_m=2, batch_size=10000, n_steps=15, seed=42):
    rng = np.random.RandomState(seed)
    n, d = X.shape
    k = len(init_centroids)
    centroids = init_centroids.copy()
    counts = np.zeros(k, dtype=np.int32)

    group_centers = centroids[rng.choice(k, m_groups, replace=False)].copy()
    diff = centroids[:, None, :] - group_centers[None, :, :]
    centroid_groups = np.argmin(np.sum(diff**2, axis=-1), axis=1)

    dist_computations = 0
    t0 = time.time()
    shuffled_idx = rng.permutation(n)

    for step in range(n_steps):
        start = (step * batch_size) % (n - batch_size)
        batch = X[shuffled_idx[start:start + batch_size]]

        b_norm = np.sum(batch**2, axis=1, keepdims=True)
        g_norm = np.sum(group_centers**2, axis=1, keepdims=True).T

        group_dists = b_norm + g_norm - 2.0 * np.dot(batch, group_centers.T)
        top_groups = np.argpartition(group_dists, top_m - 1, axis=1)[:, :top_m]
        dist_computations += len(batch) * m_groups

        group_members = [np.where(centroid_groups == g)[0] for g in range(m_groups)]

        for m_idx in range(top_m):
            for g in range(m_groups):
                c_indices = group_members[g]
                if len(c_indices) == 0:
                    continue

                pts_mask = (top_groups[:, m_idx] == g)
                pts = batch[pts_mask]
                if len(pts) == 0:
                    continue

                sub_c = centroids[c_indices]
                sub_norm = np.sum(sub_c**2, axis=1, keepdims=True).T
                pts_norm = np.sum(pts**2, axis=1, keepdims=True)

                local_dists = pts_norm + sub_norm - 2.0 * np.dot(pts, sub_c.T)
                dist_computations += len(pts) * len(c_indices)

                best_local = np.argmin(local_dists, axis=1)

                for local_id in np.unique(best_local):
                    c_id = c_indices[local_id]
                    pts_k = pts[best_local == local_id]
                    counts[c_id] += len(pts_k)
                    eta = len(pts_k) / counts[c_id]
                    centroids[c_id] = (1.0 - eta) * centroids[c_id] + eta * pts_k.mean(axis=0)

        diff = centroids[:, None, :] - group_centers[None, :, :]
        centroid_groups = np.argmin(np.sum(diff**2, axis=-1), axis=1)
        for g in range(m_groups):
            members = centroids[centroid_groups == g]
            if len(members) > 0:
                group_centers[g] = members.mean(axis=0)

    runtime = time.time() - t0
    return centroids, runtime, dist_computations


def compute_distortion(X_eval, centroids):
    c_norm = np.sum(centroids**2, axis=1, keepdims=True).T
    x_norm = np.sum(X_eval**2, axis=1, keepdims=True)
    dists = x_norm + c_norm - 2.0 * np.dot(X_eval, centroids.T)
    return float(np.mean(np.min(dists, axis=1)))


if __name__ == "__main__":
    print("loading covertype dataset for scaling analysis...")
    X_raw, _ = fetch_covtype(return_X_y=True)

    np.random.seed(42)
    perm = np.random.permutation(len(X_raw))
    X = X_raw[perm][:100000].astype(np.float32)

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X = (X - mean) / std

    n_samples, n_features = X.shape
    eval_idx = np.random.choice(n_samples, 10000, replace=False)
    X_eval = X[eval_idx]

    k_list = [500, 1000, 2000]
    m_list = [20, 32, 50]

    results = []

    for k, m in zip(k_list, m_list):
        print(f"\n--- evaluating scaling for K = {k} (M = {m}) ---")
        init_idx = np.random.choice(n_samples, k, replace=False)
        initial_centroids = X[init_idx].copy()

        c_std, t_std, ops_std = standard_kmeans(X, initial_centroids, max_iters=8)
        dist_std = compute_distortion(X_eval, c_std)

        c_mkm, t_mkm, ops_mkm = fast_mkm(X, initial_centroids, m_groups=m, top_m=2, batch_size=10000, n_steps=15)
        dist_mkm = compute_distortion(X_eval, c_mkm)

        speedup = t_std / t_mkm
        ops_cut = (1.0 - (ops_mkm / ops_std)) * 100

        results.append({
            'k': k, 'm': m,
            't_std': t_std, 't_mkm': t_mkm,
            'speedup': speedup, 'ops_cut': ops_cut,
            'dist_std': dist_std, 'dist_mkm': dist_mkm
        })

        print(f"K={k}: Standard={t_std:.2f}s, Fast MKM={t_mkm:.2f}s | Speedup: {speedup:.2f}x | Ops Cut: {ops_cut:.1f}%")

    print("\n" + "=" * 70)
    print(f"{'K Clusters':<12} | {'Std Time':<10} | {'MKM Time':<10} | {'Speedup':<10} | {'Ops Cut'}")
    print("-" * 70)
    for r in results:
        print(f"{r['k']:<12} | {r['t_std']:<8.2f}s | {r['t_mkm']:<8.2f}s | {r['speedup']:<8.2f}x | {r['ops_cut']:.1f}%")
    print("=" * 70)
