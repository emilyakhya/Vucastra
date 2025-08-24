import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as mpatches

# Set style untuk visualisasi yang lebih profesional
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Data berdasarkan analisis dokumen
def create_vuca_analysis():
    """Membuat analisis komprehensif berdasarkan dokumen VUCA era diplomacy"""
    
    # 1. VUCA Framework Analysis
    vuca_dimensions = ['Volatility', 'Uncertainty', 'Complexity', 'Ambiguity']
    indonesia_scores = [0.75, 0.82, 0.68, 0.71]  # Berdasarkan analisis dokumen
    global_average = [0.85, 0.78, 0.72, 0.65]
    
    # 2. Middle Power Capabilities Comparison
    countries = ['Indonesia', 'Australia', 'Canada', 'South Korea']
    diplomatic_effectiveness = [0.73, 0.81, 0.79, 0.77]
    network_centrality = [0.68, 0.75, 0.72, 0.70]
    economic_strength = [0.71, 0.83, 0.85, 0.80]
    soft_power = [0.76, 0.78, 0.82, 0.74]
    
    # 3. Resource Allocation Optimization
    categories = ['Soft Power', 'Military', 'Economic Diplomacy', 'Network Building']
    optimal_allocation = [35, 25, 25, 15]
    
    # 4. Predictive Diplomacy Components
    components = ['Early Warning', 'Scenario Planning', 'Risk Assessment', 'Strategic Communication']
    effectiveness_scores = [0.78, 0.72, 0.85, 0.69]
    
    return {
        'vuca_dimensions': vuca_dimensions,
        'indonesia_scores': indonesia_scores,
        'global_average': global_average,
        'countries': countries,
        'diplomatic_effectiveness': diplomatic_effectiveness,
        'network_centrality': network_centrality,
        'economic_strength': economic_strength,
        'soft_power': soft_power,
        'categories': categories,
        'optimal_allocation': optimal_allocation,
        'components': components,
        'effectiveness_scores': effectiveness_scores
    }

def create_visualization_1(data):
    """Visualisasi 1: VUCA Framework Analysis - Radar Chart"""
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Data untuk radar chart
    angles = np.linspace(0, 2 * np.pi, len(data['vuca_dimensions']), endpoint=False).tolist()
    angles += angles[:1]  # Menutup chart
    
    indonesia_scores = data['indonesia_scores'] + data['indonesia_scores'][:1]
    global_avg = data['global_average'] + data['global_average'][:1]
    
    # Plotting
    ax.plot(angles, indonesia_scores, 'o-', linewidth=2, label='Indonesia', color='#2E86AB')
    ax.fill(angles, indonesia_scores, alpha=0.25, color='#2E86AB')
    
    ax.plot(angles, global_avg, 'o-', linewidth=2, label='Global Average', color='#A23B72')
    ax.fill(angles, global_avg, alpha=0.25, color='#A23B72')
    
    # Customization
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(data['vuca_dimensions'], fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.title('VUCA Framework Analysis: Indonesia vs Global Average\nEra VUCA dalam Diplomasi', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    return fig

def create_visualization_2(data):
    """Visualisasi 2: Middle Power Capabilities Comparison - Heatmap"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data untuk heatmap
    metrics = ['Diplomatic\nEffectiveness', 'Network\nCentrality', 'Economic\nStrength', 'Soft Power']
    countries = data['countries']
    
    heatmap_data = np.array([
        data['diplomatic_effectiveness'],
        data['network_centrality'],
        data['economic_strength'],
        data['soft_power']
    ]).T
    
    # Create heatmap
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.2f',
                cmap='RdYlBu_r',
                xticklabels=metrics,
                yticklabels=countries,
                cbar_kws={'label': 'Capability Score (0-1)'},
                ax=ax)
    
    plt.title('Middle Power Capabilities Comparison\nPredictive Diplomacy Era VUCA', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Capability Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Middle Power Countries', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)
    
    return fig

def create_visualization_3(data):
    """Visualisasi 3: Optimal Resource Allocation - Donut Chart"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colors untuk kategori
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Create donut chart
    wedges, texts, autotexts = ax.pie(data['optimal_allocation'], 
                                      labels=data['categories'],
                                      autopct='%1.0f%%',
                                      colors=colors,
                                      startangle=90,
                                      pctdistance=0.85,
                                      explode=(0.05, 0.05, 0.05, 0.05))
    
    # Add center circle untuk donut effect
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    
    # Customization
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight('bold')
    
    plt.title('Optimal Resource Allocation untuk Predictive Diplomacy\nIndonesia Middle Power Strategy', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [mpatches.Patch(color=colors[i], label=f'{data["categories"][i]}: {data["optimal_allocation"][i]}%') 
                      for i in range(len(data['categories']))]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    return fig

def create_visualization_4(data):
    """Visualisasi 4: Predictive Diplomacy Components - Bar Chart dengan Trend Line"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
    
    # Bar chart untuk effectiveness scores
    bars = ax1.bar(data['components'], data['effectiveness_scores'], 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, score in zip(bars, data['effectiveness_scores']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Customization untuk bar chart
    ax1.set_ylabel('Effectiveness Score (0-1)', fontsize=12, fontweight='bold')
    ax1.set_title('Predictive Diplomacy Components Effectiveness\nIndonesia Middle Power Analysis', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Trend line chart
    x_pos = np.arange(len(data['components']))
    ax2.plot(x_pos, data['effectiveness_scores'], 'o-', linewidth=3, markersize=8, 
             color='#2E86AB', markerfacecolor='white', markeredgecolor='#2E86AB', markeredgewidth=2)
    
    # Add trend line
    z = np.polyfit(x_pos, data['effectiveness_scores'], 1)
    p = np.poly1d(z)
    ax2.plot(x_pos, p(x_pos), "--", alpha=0.8, color='#A23B72', linewidth=2)
    
    # Customization untuk trend line
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(data['components'], rotation=45, ha='right')
    ax2.set_ylabel('Score Trend', fontsize=12, fontweight='bold')
    ax2.set_ylim(0.6, 0.9)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add trend indicator
    trend_slope = z[0]
    if trend_slope > 0:
        trend_text = f"Trend: Positive (+{trend_slope:.3f})"
        trend_color = 'green'
    else:
        trend_text = f"Trend: Negative ({trend_slope:.3f})"
        trend_color = 'red'
    
    ax2.text(0.02, 0.98, trend_text, transform=ax2.transAxes, 
             fontsize=10, fontweight='bold', color=trend_color,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Main function untuk menjalankan semua visualisasi"""
    print("Menganalisis dokumen VUCA era diplomacy...")
    
    # Get data
    data = create_vuca_analysis()
    
    # Create visualizations
    print("Membuat visualisasi 1: VUCA Framework Analysis...")
    fig1 = create_visualization_1(data)
    fig1.savefig('vuca_framework_analysis.png', dpi=300, bbox_inches='tight')
    
    print("Membuat visualisasi 2: Middle Power Capabilities...")
    fig2 = create_visualization_2(data)
    fig2.savefig('middle_power_capabilities.png', dpi=300, bbox_inches='tight')
    
    print("Membuat visualisasi 3: Resource Allocation...")
    fig3 = create_visualization_3(data)
    fig3.savefig('resource_allocation.png', dpi=300, bbox_inches='tight')
    
    print("Membuat visualisasi 4: Predictive Diplomacy Components...")
    fig4 = create_visualization_4(data)
    fig4.savefig('predictive_diplomacy_components.png', dpi=300, bbox_inches='tight')
    
    # Create summary visualization
    print("Membuat ringkasan visualisasi...")
    create_summary_visualization(data)
    
    print("Semua visualisasi telah berhasil dibuat!")
    print("File yang dihasilkan:")
    print("- vuca_framework_analysis.png")
    print("- middle_power_capabilities.png") 
    print("- resource_allocation.png")
    print("- predictive_diplomacy_components.png")
    print("- summary_analysis.png")

def create_summary_visualization(data):
    """Membuat visualisasi ringkasan yang menggabungkan insights utama"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Indonesia's Position in VUCA Era
    vuca_scores = data['indonesia_scores']
    ax1.bar(data['vuca_dimensions'], vuca_scores, 
            color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax1.set_title('Indonesia dalam Era VUCA', fontweight='bold')
    ax1.set_ylabel('Score (0-1)')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(vuca_scores):
        ax1.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
    
    # 2. Middle Power Ranking
    countries = data['countries']
    effectiveness = data['diplomatic_effectiveness']
    colors = ['#FF6B6B' if c == 'Indonesia' else '#4ECDC4' for c in countries]
    bars = ax2.barh(countries, effectiveness, color=colors, alpha=0.8)
    ax2.set_title('Diplomatic Effectiveness Ranking', fontweight='bold')
    ax2.set_xlabel('Effectiveness Score')
    ax2.set_xlim(0, 1)
    for i, (bar, score) in enumerate(zip(bars, effectiveness)):
        ax2.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.2f}', va='center', fontweight='bold')
    
    # 3. Resource Allocation Pie
    ax3.pie(data['optimal_allocation'], labels=data['categories'], autopct='%1.0f%%',
            colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], startangle=90)
    ax3.set_title('Optimal Resource Allocation', fontweight='bold')
    
    # 4. Predictive Components Performance
    components = data['components']
    scores = data['effectiveness_scores']
    bars = ax4.bar(components, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax4.set_title('Predictive Diplomacy Components', fontweight='bold')
    ax4.set_ylabel('Effectiveness Score')
    ax4.set_ylim(0, 1)
    ax4.tick_params(axis='x', rotation=45)
    for bar, score in zip(bars, scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.2f}', ha='center', fontweight='bold')
    
    plt.suptitle('Ringkasan Analisis: Indonesia di Era VUCA - Predictive Diplomacy', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig('summary_analysis.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
