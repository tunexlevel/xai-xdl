import matplotlib.pyplot as plt

# Your actual results
k_labels = ['Top-1', 'Top-3', 'Top-5', 'Top-10']
accuracies = [57.00, 71.00, 73.00, 74.00]

# Setup Plot
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(k_labels, accuracies, color='#4c72b0', width=0.6, zorder=3)

# Styling
ax.set_ylim(40, 85) # Zoom in to show the progression
ax.set_ylabel('Accuracy (%)', fontsize=12)
# ax.set_title('Beam Search Performance on USPTO-50k', fontsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# Add Line to show trend
x_coords = [i for i, _ in enumerate(k_labels)]
ax.plot(x_coords, accuracies, color='red', marker='o', linewidth=2, linestyle='-', alpha=0.7, label='Cumulative Improvement')

# Add Labels on Bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.legend()
plt.tight_layout()
plt.savefig('final_top_k_results.svg', format='svg')
plt.show()