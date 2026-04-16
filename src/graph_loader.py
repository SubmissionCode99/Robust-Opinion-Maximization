import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import time
import os

def process_snap_graph(input_gz_path, output_npz_path, is_undirected=False):
    print(f"{'='*60}\nProcessing {input_gz_path}...")
    start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    
    # 1. Read edge list directly from the .gz file using Pandas
    # SNAP graphs use spaces/tabs and use '#' for comments
    edges = pd.read_csv(input_gz_path, sep=r'\s+', comment='#', header=None, names=['source', 'target'])
    
    # Map raw node IDs to contiguous integers (0 to N-1)
    unique_nodes = pd.unique(edges[['source', 'target']].values.ravel('K'))
    node_mapping = {raw_id: new_id for new_id, raw_id in enumerate(unique_nodes)}
    
    edges['source'] = edges['source'].map(node_mapping)
    edges['target'] = edges['target'].map(node_mapping)
    
    n_total = len(unique_nodes)
    print(f"  Raw graph loaded: {n_total:,} nodes, {len(edges):,} edges.")
    
    # 2. Create Sparse Adjacency Matrix
    row = edges['source'].values
    col = edges['target'].values
    data = np.ones(len(edges))
    
    A_raw = sp.coo_matrix((data, (row, col)), shape=(n_total, n_total)).tocsr()
    
    # Symmetrize if the graph is undirected (to prevent false LSCC fracturing)
    if is_undirected:
        print("  Graph is undirected. Symmetrizing adjacency matrix...")
        A_raw = A_raw + A_raw.T
        # Force all non-zero entries back to 1.0 (in case of duplicate bidirectional edges)
        A_raw.data = np.ones_like(A_raw.data)
    
    # 3. Find the Largest Strongly Connected Component (LSCC)
    print("  Extracting Largest Strongly Connected Component (LSCC)...")
    n_components, labels = connected_components(csgraph=A_raw, directed=True, connection='strong')
    
    # Find the largest component label
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_component_label = unique_labels[np.argmax(counts)]
    
    # Create a boolean mask for nodes in the LSCC
    lscc_mask = (labels == largest_component_label)
    
    # Slice the sparse matrix to keep only LSCC nodes
    A_lscc = A_raw[lscc_mask, :][:, lscc_mask]
    n_lscc = A_lscc.shape[0]
    print(f"  LSCC extracted: {n_lscc:,} nodes, {A_lscc.nnz:,} edges.")
    
    # 4. Row-Normalize to create W
    print("  Row-normalizing to create interaction matrix W...")
    row_sums = np.array(A_lscc.sum(axis=1)).flatten()
    
    # Prevent division by zero (should not happen in LSCC, but good practice)
    row_sums[row_sums == 0] = 1.0 
    inv_D = sp.diags(1.0 / row_sums)
    
    W_sparse = inv_D @ A_lscc
    
    # 5. Save as heavily compressed .npz
    sp.save_npz(output_npz_path, W_sparse)
    print(f"  Saved cleanly to {output_npz_path}.")
    print(f"  Total time: {time.time() - start_time:.2f} seconds.\n")

if __name__ == "__main__":
    # Define the input/output pairs and their directed/undirected status
    datasets = [
        ("data/raw/soc-Epinions1.txt.gz", "data/processed/W_epinions.npz", False),
        ("data/raw/soc-Slashdot.txt.gz", "data/processed/W_slashdot.npz", False),
        ("data/raw/loc-gowalla_edges.txt.gz", "data/processed/W_gowalla.npz", True),
        ("data/raw/web-Google.txt.gz", "data/processed/W_google.npz", False),
        ("data/raw/soc-pokec-relationships.txt.gz", "data/processed/W_pokec.npz", False),
    ]

    for in_path, out_path, is_undir in datasets:
        if os.path.exists(in_path):
            process_snap_graph(in_path, out_path, is_undirected=is_undir)
        else:
            print(f"File not found: {in_path}. Please ensure it is in the data/raw/ folder.")