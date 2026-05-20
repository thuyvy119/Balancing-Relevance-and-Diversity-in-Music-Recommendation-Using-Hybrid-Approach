import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from datetime import datetime

def visualize_evaluation_results(relevance_metrics, diversity_metrics, query="", save_plots=True, save_dir="evaluation_plots"):
    """
    Main evaluation visualization with save option
    """
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create save directory if needed
    if save_plots and save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Create the main plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Recommendation System Evaluation - {query}', fontsize=16, fontweight='bold')
    
    # if not relevance_metrics or not diversity_metrics:
    #     print("No evaluation metrics available for visualization.")
    # return []
    
    # Convert to DataFrames
    rel_df = pd.DataFrame(relevance_metrics)
    div_df = pd.DataFrame(diversity_metrics)
    
    # 1. Relevance vs Diversity Trade-off
    scatter = axes[0].scatter(
        rel_df['Precision@10'], 
        div_df['Serendipity'], 
        alpha=0.7, 
        s=80, 
        c=rel_df['MAP@10'], 
        cmap='viridis',
        edgecolors='white',
        linewidth=0.5
    )
    axes[0].set_xlabel('Precision@10 (Relevance)', fontsize=12)
    axes[0].set_ylabel('Serendipity (Diversity)', fontsize=12)
    axes[0].set_title('Relevance vs Diversity Trade-off\n(colored by MAP@10)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Add colorbar for MAP
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label('MAP@10', fontsize=10)
    
    # 2. Key Metrics Summary
    key_metrics = {
        'Precision': rel_df['Precision@10'].mean(),
        'MAP': rel_df['MAP@10'].mean(),
        'MRR': rel_df['MRR@10'].mean(),
        'Similarity': rel_df['avg_similarity'].mean(),
        'Novelty': div_df['Novelty'].mean(),
        'Serendipity': div_df['Serendipity'].mean(),
        'Coverage': div_df['CategoryCoverage'].mean(),
        'ILS': div_df['IntraListSimilarity'].mean()
    }
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3BB273', '#7768AE', '#5B6C5D', '#E3B505']
    bars = axes[1].bar(key_metrics.keys(), key_metrics.values(), color=colors, alpha=0.8)
    axes[1].set_title('Key Performance Metrics', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, key_metrics.values()):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Set y-axis limit to accommodate labels
    max_val = max(key_metrics.values())
    axes[1].set_ylim(0, max_val * 1.15)
    
    plt.tight_layout()
    
    # Save the plot if requested
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_slug = query.replace(' ', '_').replace('-', '_')[:40]
        filename = f"{timestamp}_{query_slug}_evaluation.png"
        filepath = os.path.join(save_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Evaluation plot saved to: {filepath}")
    
    plt.show()
    
    return key_metrics


def create_comprehensive_visualizations(relevance_metrics, diversity_metrics, query="", save_dir="evaluation_plots"):
    """
    Comprehensive visualizations: Main plot + Distribution plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Main evaluation plot
    key_metrics = visualize_evaluation_results(relevance_metrics, diversity_metrics, query, True, save_dir)
    
    # Distribution plots
    create_distribution_plots(relevance_metrics, diversity_metrics, query, save_dir)
    
    return key_metrics

def create_distribution_plots(relevance_metrics, diversity_metrics, query_name="Query", save_dir="evaluation_plots"):
    """Create distribution plots for key metrics"""
    rel_df = pd.DataFrame(relevance_metrics)
    div_df = pd.DataFrame(diversity_metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Metric Distributions - {query_name}', fontsize=16, fontweight='bold')
    
    # Key metrics to plot
    metrics_to_plot = [
        ('Precision@10', rel_df['Precision@10'], 'skyblue', 'Precision Distribution'),
        ('MAP@10', rel_df['MAP@10'], 'lightgreen', 'MAP Distribution'), 
        ('Novelty', div_df['Novelty'], 'orange', 'Novelty Distribution'),
        ('CategoryCoverage', div_df['CategoryCoverage'], 'purple', 'Category Coverage Distribution')
    ]
    
    for idx, (metric_name, data, color, title) in enumerate(metrics_to_plot):
        row, col = idx // 2, idx % 2
        axes[row, col].hist(data, bins=20, alpha=0.7, color=color, edgecolor='black')
        axes[row, col].set_title(title)
        axes[row, col].set_xlabel(metric_name)
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].axvline(data.mean(), color='red', linestyle='--', 
                            label=f'Mean: {data.mean():.3f}')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save distribution plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_slug = query_name.replace(' ', '_').replace('-', '_')[:40]
    filename = f"{save_dir}/{timestamp}_{query_slug}_distributions.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f" Distribution plot saved to: {filename}")
    
    plt.show()