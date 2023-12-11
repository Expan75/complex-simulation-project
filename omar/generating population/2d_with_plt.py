import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
class PartyDataGenerator:
    def __init__(self, data, total_samples=10000):
        self.df = pd.DataFrame(data)
        self.total_samples = total_samples
        self.scaler = MinMaxScaler()
        self.scaled_features = self.scaler.fit_transform(self.df[['lrecon', 'galtan']])
        self.vote_percentages = self.df['vote_percentage'].values
        self.synthetic_data = self.generate_synthetic_data()

    def generate_synthetic_data(self):
        synthetic_data = np.empty((0, 2))
        for i, party in self.df.iterrows():
            num_samples = int(party['vote_percentage'] / 100 * self.total_samples)
            samples = np.random.multivariate_normal(
                mean=self.scaled_features[i],
                cov=np.diag(np.full(self.scaled_features.shape[1], 0.05)),
                size=num_samples
            )
            synthetic_data = np.vstack([synthetic_data, samples])
        return synthetic_data

    def assign_to_nearest_party(self):
        assignments = []
        for point in self.synthetic_data:
            distances = np.linalg.norm(self.scaled_features - point, axis=1)
            nearest_party = np.argmin(distances)
            assignments.append(nearest_party)
        return np.array(assignments)

original_data = {
    'party': ['V', 'S/SAP', 'C', 'L', 'M', 'KD', 'MP', 'SD'],
    'lrecon': [1.764706, 4.117647, 8.000000, 7.117647, 7.705883, 7.235294, 3.941176, 5.588235],
    'galtan': [1.941176, 4.411765, 2.235294, 3.235294, 5.941176, 7.058824, 1.588235, 8.764706],
    'vote_percentage': [15, 20, 10, 25, 10, 5, 10, 5]
}

data_generator = PartyDataGenerator(original_data)
scaled_features = data_generator.scaled_features
vote_percentages = data_generator.vote_percentages
synthetic_data = data_generator.synthetic_data

synthetic_assignments = data_generator.assign_to_nearest_party()

df = pd.DataFrame(original_data)



num_components = len(df['party'])
total_samples = data_generator.total_samples

colors = sns.color_palette('hsv', num_components) 

fig, ax = plt.subplots(figsize=(12, 8))

for i in range(num_components):
    party_data = synthetic_data[synthetic_assignments == i]
    ax.scatter(party_data[:, 0], party_data[:, 1], color=colors[i], alpha=0.2)

for i, (x, y) in enumerate(scaled_features):
    ax.scatter(x, y, color='red', edgecolor='k', zorder=5)
    ax.text(x, y, str(i+1), color='white', fontsize=12, weight='bold', ha='center', va='center',
            bbox=dict(facecolor=colors[i], edgecolor='none', pad=2, alpha=0.8), zorder=5)

from matplotlib.offsetbox import AnchoredText
party_texts = [f"{party} ({vote}%) #{i+1}" for i, (party, vote) in enumerate(zip(original_data['party'], original_data['vote_percentage']))]
at = AnchoredText("\n".join(party_texts),
                  prop=dict(size=12), frameon=True,
                  loc='upper left',
                  bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
at.patch.set_boxstyle("round,pad=0.5,rounding_size=0.2")
ax.add_artist(at)

plt.title('Electorate Distribution with Party Positions')
plt.xlabel('LRECON Scaled')
plt.ylabel('GALTAN Scaled')
plt.tight_layout()
plt.show()