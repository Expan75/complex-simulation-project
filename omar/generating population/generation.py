from population import PartyDataVisualizer
from data_ret import *
party_data = PartyData('research/CHES2019V3.csv', 16)

partiesData = PartyDataVisualizer(party_data.get_dataframe())

population = partiesData.generate_population()
print(population)