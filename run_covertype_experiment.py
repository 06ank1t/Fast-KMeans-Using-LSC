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
        # process in chunks to keep memory usage low
        chunk_size = 5000
        labels = np.empty(n, dtype=np.int32)
        c_norm = np.sum(centroids**2, axis=1, keepdims=True).T

        for start in range(0, n, chunk_size):
            chunk = X[start:start + chunk_size]
            x_norm = np.sum(chunk**2, axis=1, keepdims=True)
            dists = x_norm + c_norm - 2.0 * np.dot(chunk, centroids.T)
            labels[start:start + chunk_size] = np.argmin(dists, axis=1)
            dist_computations += len(chunk) * k

        # recompute centroids from cluster assignments
        counts = np.bincount(labels, minlength=k)
        sums = np.zeros((k, d), dtype=np.float32)
        np.add.at(sums, labels, X)
        nonzero = counts > 0
        centroids[nonzero] = sums[nonzero] / counts[nonzero, None]

    runtime = time.time() - t0
    return centroids, runtime, dist_computations


def minibatch_kmeans(X, init_centroids, batch_size=10000, n_steps=15, seed=42):
    rng = np.random.RandomState(seed)
    n, d = X.shape
    k = len(init_centroids)
    centroids = init_centroids.copy()
    counts = np.zeros(k, dtype=np.int32)

    dist_computations = 0
    t0 = time.time()

    for step in range(n_steps):
        # sample a random batch
        batch_idx = rng.choice(n, batch_size, replace=False)
        batch = X[batch_idx]

        c_norm = np.sum(centroids**2, axis=1, keepdims=True).T
        b_norm = np.sum(batch**2, axis=1, keepdims=True)

        dists = b_norm + c_norm - 2.0 * np.dot(batch, centroids.T)
        labels = np.argmin(dists, axis=1)
        dist_computations += len(batch) * k

        # update centroids using running average
        for local_id in np.unique(labels):
            pts = batch[labels == local_id]
            counts[local_id] += len(pts)
            eta = len(pts) / counts[local_id]
            centroids[local_id] = (1.0 - eta) * centroids[local_id] + eta * pts.mean(axis=0)

    runtime = time.time() - t0
    return centroids, runtime, dist_computations


def fast_mkm(X, init_centroids, m_groups=32, top_m=2, batch_size=10000, n_steps=15, seed=42):
    rng = np.random.RandomState(seed)
    n, d = X.shape
    k = len(init_centroids)
    centroids = init_centroids.copy()
    counts = np.zeros(k, dtype=np.int32)

    # cluster centroids into M groups first
    group_centers = centroids[rng.choice(k, m_groups, replace=False)].copy()
    diff = centroids[:, None, :] - group_centers[None, :, :]
    centroid_groups = np.argmin(np.sum(diff**2, axis=-1), axis=1)

    dist_computations = 0
    t0 = time.time()

    # shuffle data once so we can iterate sequentially in batches
    shuffled_idx = rng.permutation(n)

    for step in range(n_steps):
        start = (step * batch_size) % (n - batch_size)
        batch = X[shuffled_idx[start:start + batch_size]]

        b_norm = np.sum(batch**2, axis=1, keepdims=True)
        g_norm = np.sum(group_centers**2, axis=1, keepdims=True).T

        # coarse search: compare batch only to group centroids
        group_dists = b_norm + g_norm - 2.0 * np.dot(batch, group_centers.T)
        top_groups = np.argpartition(group_dists, top_m - 1, axis=1)[:, :top_m]
        dist_computations += len(batch) * m_groups

        # fine search: only check centroids belonging to top groups
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

        # update group centers for next iteration
        diff = centroids[:, None, :] - group_centers[None, :, :]
        centroid_groups = np.argmin(np.sum(diff**2, axis=-1), axis=1)
        for g in range(m_groups):
            members = centroids[centroid_groups == g]
            if len(members) > 0:
                group_centers[g] = members.mean(axis=0)

    runtime = time.time() - t0
    return centroids, runtime, dist_computations


def compute_distortion(X_eval, centroids):
    # compute average squared euclidean distance
    c_norm = np.sum(centroids**2, axis=1, keepdims=True).T
    x_norm = np.sum(X_eval**2, axis=1, keepdims=True)
    dists = x_norm + c_norm - 2.0 * np.dot(X_eval, centroids.T)
    return float(np.mean(np.min(dists, axis=1)))


if __name__ == "__main__":
    print("loading covertype dataset...")
    X_raw, _ = fetch_covtype(return_X_y=True)

    # shuffle rows so classes are uniformly distributed
    np.random.seed(42)
    perm = np.random.permutation(len(X_raw))
    X = X_raw[perm][:100000].astype(np.float32)

    # zero-mean unit-variance
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X = (X - mean) / std

    n_samples, n_features = X.shape
    k_clusters = 1000
    m_groups = 32
    batch_size = 10000
    n_steps = 15

    # same starting centroids for fair comparison
    init_idx = np.random.choice(n_samples, k_clusters, replace=False)
    initial_centroids = X[init_idx].copy()

    # validation sample for distortion
    eval_idx = np.random.choice(n_samples, 10000, replace=False)
    X_eval = X[eval_idx]

    print(f"data ready: {n_samples} samples, {n_features} features, k={k_clusters}")

    print("\nrunning standard kmeans...")
    c_std, t_std, ops_std = standard_kmeans(X, initial_centroids, max_iters=8)
    dist_std = compute_distortion(X_eval, c_std)
    print(f"standard done in {t_std:.2f}s | distortion: {dist_std:.3f}")

    print("\nrunning mini-batch kmeans...")
    c_mb, t_mb, ops_mb = minibatch_kmeans(X, initial_centroids, batch_size=batch_size, n_steps=n_steps)
    dist_mb = compute_distortion(X_eval, c_mb)
    print(f"mini-batch done in {t_mb:.2f}s | distortion: {dist_mb:.3f}")

    print("\nrunning fast mkm...")
    c_mkm, t_mkm, ops_mkm = fast_mkm(X, initial_centroids, m_groups=m_groups, top_m=2, batch_size=batch_size, n_steps=n_steps)
    dist_mkm = compute_distortion(X_eval, c_mkm)
    print(f"fast mkm done in {t_mkm:.2f}s | distortion: {dist_mkm:.3f}")

    # summary table
    print("\n" + "-" * 62)
    print(f"{'Method':<24} | {'Time':<8} | {'Dist Evals':<13} | {'Distortion'}")
    print("-" * 62)
    print(f"{'Standard Lloyd':<24} | {t_std:.2f}s   | {ops_std:<13,} | {dist_std:.3f}")
    print(f"{'Mini-Batch':<24} | {t_mb:.2f}s   | {ops_mb:<13,} | {dist_mb:.3f}")
    print(f"{'Multi-Stage MKM':<24} | {t_mkm:.2f}s   | {ops_mkm:<13,} | {dist_mkm:.3f}")
    print("-" * 62)
    print(f"MKM vs Standard Speedup   : {t_std / t_mkm:.2f}x faster")
    print(f"MKM vs Mini-Batch Speedup : {t_mb / t_mkm:.2f}x faster")
    print(f"Distance operations cut   : {(1.0 - ops_mkm/ops_std)*100:.1f}% reduction")
