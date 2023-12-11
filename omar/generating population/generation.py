from population import PartyDataVisualizer
from data_ret import *
import numpy as np
party_data = PartyData('research/CHES2019V3.csv', 16)

partiesData = PartyDataVisualizer(party_data.get_dataframe())

population = partiesData.generate_population()


"""
The swedish parties in the study
V
S/SAP
C
L
M
KD
MP
SD 
"""
np.savetxt("pop.csv", population, delimiter=",", header='V, S, C,L,M,KD,MP,SD', comments='')


print(type(population))