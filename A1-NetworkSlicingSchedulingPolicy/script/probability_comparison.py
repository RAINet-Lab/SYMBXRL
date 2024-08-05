import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def kl_divergence(p, q):
    """
    Calculate the Kullback-Leibler divergence between two discrete probability distributions.
    Both p and q must be numpy arrays of the same length, and q must not contain zeros.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Normalize p and q to ensure they sum to 1
    p /= p.sum()
    q /= q.sum()
    
    # Replace zeros with a very small number to avoid division by zero
    q = np.where(q == 0, 1e-10, q)
    p = np.where(p == 0, 1e-10, p)
    
    # Calculate KL divergence
    divergence = np.sum(np.where(p != 0, p * np.log(p / q), 0))
    
    # Ensure divergence is not negative due to numerical issues
    return max(divergence, 0)

def js_divergence(p, q):
    """
    Calculate the Jensen-Shannon divergence between two discrete probability distributions.
    Both p and q must be numpy arrays of the same length.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Normalize p and q to ensure they sum to 1
    p /= p.sum()
    q /= q.sum()
    
    # Calculate the pointwise mean of p and q
    m = 0.5 * (p + q)
    
    # Calculate the Jensen-Shannon divergence using the KL divergence
    js_divergence = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    
    return js_divergence

def wasserstein_distance(p, q):
    """
    Calculate the Wasserstein distance (Earth Mover's distance) between two one-dimensional discrete probability distributions.
    Both p and q must be numpy arrays of the same length.
    """
    # Normalize p and q to ensure they sum to 1
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    
    # Calculate the cumulative distribution functions (CDFs) of p and q
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    
    # Calculate the Wasserstein distance as the L1 distance between the CDFs
    distance = np.sum(np.abs(cdf_p - cdf_q))    
    
    return distance

def hellinger_distance(p, q):
    """
    Calculate the Hellinger distance between two discrete probability distributions.
    Both p and q must be numpy arrays of the same length.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Normalize p and q to ensure they sum to 1
    p /= p.sum()
    q /= q.sum()
    
    # Calculate the Hellinger distance
    distance = np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
    
    return distance

def mahalanobis_distance(p, q):
    """
    Calculate the Mahalanobis distance between two discrete probability distributions.
    Both p and q must be numpy arrays of the same length.
    The covariance matrix is computed from the combined array of p and q.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Normalize p and q to ensure they sum to 1
    p /= p.sum()
    q /= q.sum()
    
    # Calculate the difference vector
    diff = p - q
    
    # Combine p and q to compute the covariance matrix
    combined = np.vstack([p, q])
    cov = np.cov(combined, rowvar=False)  # Ensure that rows represent variables
    
    # Invert the covariance matrix
    inv_cov = np.linalg.inv(cov)
    
    # Calculate the Mahalanobis distance
    distance = np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))
    
    return distance


def plot_and_save_probability_dist_comparison_heatmaps_with_clustering(probs, unique_ids, name, path, methods=["hd"]):
    """
    This function will calculate and plot the probability distribution comparison for probability distribution a and b using 
    methods listed in the methods array amd save them in path with the name given
    This function can be only applied to symmetric methods. So fot this method that we want to apply clustering we will only use JS and Hellinger method
    """
    for method in methods:
        div_matrix = np.zeros((len(unique_ids), len(unique_ids)))
        if method == "js":
            ## Calculate comparison value
            for i, uid_i in enumerate(unique_ids):
                for j, uid_j in enumerate(unique_ids):
                    if i != j:
                        prob_i = probs[probs['id'] == uid_i]['prob'].values[0]
                        prob_j = probs[probs['id'] == uid_j]['prob'].values[0]
                        divergence = js_divergence(prob_i, prob_j)
                        div_matrix[i, j] = divergence
                        div_matrix[j, i] = divergence  # Ensure the matrix is symmetric
                    else:
                        div_matrix[i, j] = 0  # Diagonal should be zero
            # Perform hierarchical clustering
            linkage_matrix = linkage(squareform(div_matrix), method='average')
            sorted_indices = leaves_list(linkage_matrix)
            
            # Reorder the matrix according to the clustering
            sorted_div_matrix = div_matrix[:, sorted_indices][sorted_indices, :]
            
            # Plot the comparison
            div_df = pd.DataFrame(sorted_div_matrix, index=unique_ids[sorted_indices], columns=unique_ids[sorted_indices])
            fig = px.imshow(
                div_df,
                labels=dict(x="ID", y="ID", color=f"{method} divergence"),
                title=f"{name} - {method} divergence",
                range_color=[0, 1]
            )
            fig.update_layout(
                title=f'Heatmap of {method.capitalize()} Divergence between Distributions',
                xaxis_title='ID',
                yaxis_title='ID',
                autosize=True,
            )
            # Save the plot for each combination of agent and number of users
            full_file_path = os.path.join(path, f"{name}_{method}_divergence.html")
            fig.write_html(full_file_path)
        elif method == "hd":
            ## Calculate comparison value
            for i, uid_i in enumerate(unique_ids):
                for j, uid_j in enumerate(unique_ids):
                    if i != j:
                        prob_i = probs[probs['id'] == uid_i]['prob'].values[0]
                        prob_j = probs[probs['id'] == uid_j]['prob'].values[0]
                        divergence = hellinger_distance(prob_i, prob_j)
                        div_matrix[i, j] = divergence
                        div_matrix[j, i] = divergence  # Ensure the matrix is symmetric
                    else:
                        div_matrix[i, j] = 0  # Diagonal should be zero
            # Perform hierarchical clustering
            linkage_matrix = linkage(squareform(div_matrix), method='average')
            sorted_indices = leaves_list(linkage_matrix)
            
            # Reorder the matrix according to the clustering
            sorted_div_matrix = div_matrix[:, sorted_indices][sorted_indices, :]
            
            # Plot the comparison
            div_df = pd.DataFrame(sorted_div_matrix, index=unique_ids[sorted_indices], columns=unique_ids[sorted_indices])
            fig = px.imshow(
                div_df,
                labels=dict(x="ID", y="ID", color=f"Hellinger Distance"),
                title=f"{name} - Hellinger Distance",
                range_color=[0, 1]
            )
            fig.update_layout(
                title=f'Heatmap of Hellinger Distance between Distributions',
                xaxis_title='ID',
                yaxis_title='ID',
                autosize=True,
            )
            # Save the plot for each combination of agent and number of users
            full_file_path = os.path.join(path, f"{name}_{method}_divergence.html")
            fig.write_html(full_file_path)

