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
"""
rows

'lrecon','lrecon_sd','lrecon_salience'
                   ,'lrecon_dissent','lrecon_blur','galtan','galtan_sd',
                   'galtan_salience','galtan_dissent','galtan_blur','immigrate_policy',
                   'immigrate_salience','immigrate_dissent','multiculturalism','multicult_salience',
                   'multicult_dissent','redistribution','redist_salience','environment',
                   'enviro_salience','spendvtax','deregulation','econ_interven',
                   'civlib_laworder','sociallifestyle','religious_principles','ethnic_minorities'
                   ,'nationalism','urban_rural','protectionism','regions',
                   'russian_interference','anti_islam_rhetoric','people_vs_elite','antielite_salience',
                   'corrupt_salience','members_vs_leadership'

"""
np.savetxt("pop.csv", population, delimiter=",", header='V, S, C,L,M,KD,MP,SD', comments='')


print(type(population))