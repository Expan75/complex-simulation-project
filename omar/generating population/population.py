import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from data_ret import * 


class PartyDataVisualizer:
    """
    A class to visualize party data in an interactive scatter plot.

    Attributes:
        df (pd.DataFrame): DataFrame of the data.
        scaler (MinMaxScaler): Scaler for normalizing data.
        scaled_features (np.ndarray): Scaled features of the data.
        vote_percentages (np.ndarray): Vote percentages.
        colors (list): Color palette for the parties.
        total_samples (int):Number of individuals to generate.
        population (np.ndarray): Synthetic population data.
        population_votes (np.ndarray): Assigned party for each individual in the population.
    """
    def __init__(self, data: dict, num_individuals: int = 10000, make_plot: bool = False):
        self.df = data
        self.scaler = MinMaxScaler()
        self.scaled_features = self.scaler.fit_transform(self.df[['lrecon','lrecon_sd','lrecon_salience'
                   ,'lrecon_dissent','lrecon_blur','galtan','galtan_sd','galtan_salience','galtan_dissent','galtan_blur','immigrate_policy','immigrate_salience'
                   ,'immigrate_dissent','multiculturalism','multicult_salience','multicult_dissent','redistribution'
                   ,'redist_salience','environment','enviro_salience','spendvtax','deregulation',
                   'econ_interven','civlib_laworder','sociallifestyle','religious_principles',
                   'ethnic_minorities','nationalism','urban_rural','protectionism','regions','russian_interference','anti_islam_rhetoric','people_vs_elite','antielite_salience','corrupt_salience','members_vs_leadership']])
        self.vote_percentages = self.df['vote_percentage'].values
        self.colors = sns.color_palette('hsv', len(self.df['party']))
        self.total_samples = num_individuals
        self.population = self.generate_population()
        if make_plot == True:
            self.population_votes = self.assign_to_nearest_party()
            self.setup_gui()

    def generate_population(self) -> np.ndarray:
        population = np.empty((0, self.scaled_features.shape[1]))  
        for i, party in self.df.iterrows():
            num_samples = int(party['vote_percentage'] / 100 * self.total_samples)
            samples = np.random.multivariate_normal(
                mean=self.scaled_features[i],
                cov=np.diag(np.full(self.scaled_features.shape[1], 0.05)),
                size=num_samples
            )
            population = np.vstack([population, samples])
        return population

    def assign_to_nearest_party(self) -> np.array:
        assignments = []
        for point in self.population:
            distances = np.linalg.norm(self.scaled_features - point, axis=1)
            nearest_party = np.argmin(distances)
            assignments.append(nearest_party)
        return np.array(assignments)

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Interactive Scatter Plot")

        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, columnspan=4)

        self.x_var = tk.StringVar(self.root)
        self.x_dropdown = ttk.Combobox(self.root, textvariable=self.x_var, values=list(self.df.columns[1:]))
        self.x_dropdown.grid(row=1, column=0)
        self.x_dropdown.current(0)  

        self.y_var = tk.StringVar(self.root)
        self.y_dropdown = ttk.Combobox(self.root, textvariable=self.y_var, values=list(self.df.columns[1:]))
        self.y_dropdown.grid(row=1, column=1)
        self.y_dropdown.current(1)  

        self.update_button = tk.Button(self.root, text="Update Plot", command=self.update_plot)
        self.update_button.grid(row=1, column=2)

        self.update_plot()

    def update_plot(self):
        selected_x = self.x_var.get()
        selected_y = self.y_var.get()
        self.scaled_features = self.scaler.fit_transform(self.df[[selected_x, selected_y]])
        self.population = self.generate_population()
        self.population_votes = self.assign_to_nearest_party()

        self.ax.clear()
        for i in range(len(self.df['party'])):
            party_data = self.population[self.population_votes == i]
            self.ax.scatter(party_data[:, 0], party_data[:, 1], color=self.colors[i], alpha=0.5, label=f'#{i+1}')

        for i, (x, y) in enumerate(self.scaled_features):
            self.ax.scatter(x, y, color='red', edgecolor='k', zorder=5)
            self.ax.text(x, y, str(i+1), color='white', fontsize=12, weight='bold', ha='center', va='center',
                        bbox=dict(facecolor=self.colors[i], edgecolor='none', pad=2, alpha=0.8), zorder=5)

        party_texts = [f"{party} ({vote}%) #{i+1}" for i, (party, vote) in enumerate(zip(self.df['party'], self.df['vote_percentage']))]
        at = AnchoredText("\n".join(party_texts),
                        prop=dict(size=12), frameon=True,
                        loc='upper left',
                        bbox_to_anchor=(0.95, 1), bbox_transform=self.ax.transAxes)
        at.patch.set_boxstyle("round,pad=0.5,rounding_size=0.2")
        self.ax.add_artist(at)
        self.ax.set_title('Electorate Distribution with Party Positions')
        self.ax.set_xlabel(selected_x + ' Scaled')
        self.ax.set_ylabel(selected_y + ' Scaled')

        self.canvas.draw()


if __name__ == "__main__":
    party_data = PartyData('research/CHES2019V3.csv', 16)


    visualizer = PartyDataVisualizer(party_data.get_dataframe(),make_plot=True)
    visualizer.root.mainloop()