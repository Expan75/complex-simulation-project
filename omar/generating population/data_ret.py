import pandas as pd
from typing import List

class PartyData:
    """Class for processing and storing political party data."""

    def __init__(self, filepath: str, country_code: int):
        self.filepath = filepath
        self.country_code = country_code
        self.df = None
        self._load_data()
        self._filter_by_country()
        self._select_columns()
        self.df['vote_percentage'] = [15, 20, 10, 25, 10, 5, 10, 5]  


    def _load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.filepath)

    def _filter_by_country(self):
        self.df = self.df[self.df['country'] == self.country_code]

    def _select_columns(self):
        columns_to_keep = [
                    'party', 'lrecon','lrecon_sd','lrecon_salience'
                   ,'lrecon_dissent','lrecon_blur','galtan','galtan_sd',
                   'galtan_salience','galtan_dissent','galtan_blur','immigrate_policy',
                   'immigrate_salience','immigrate_dissent','multiculturalism','multicult_salience',
                   'multicult_dissent','redistribution','redist_salience','environment',
                   'enviro_salience','spendvtax','deregulation','econ_interven',
                   'civlib_laworder','sociallifestyle','religious_principles','ethnic_minorities'
                   ,'nationalism','urban_rural','protectionism','regions',
                   'russian_interference','anti_islam_rhetoric','people_vs_elite','antielite_salience',
                   'corrupt_salience','members_vs_leadership']
        self.df = self.df[columns_to_keep]

    def set_vote_percentages(self, vote_perc: List[int]):
        if len(vote_perc) == len(self.df):
            self.df['vote_percentage'] = vote_perc
        else:
            print("Error: The length of vote percentages does not match the number of parties.")
            
    def get_dataframe(self)-> pd.DataFrame:
        self.df = self.df.reset_index(drop=True)
        return self.df

    def __str__(self) -> str:


        return str(self.df)

if __name__ == "__main__":

    party_data = PartyData('research/CHES2019V3.csv', 16)
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
    print(type(party_data.get_dataframe()))
