"""
Database
========

This module contains the Database class, which is used to generate or load a database. 
We use the [freqrir]{@link https://github.com/woodRock/freqRIR} package to perform the
RIR computations in the frequency domain.
"""

import os.path
import numpy as np
from tqdm import tqdm 
import skorch
from freqrir.freqrir import frequency_rir 
from freqrir.helper import sample_random_receiver_locations

class Database():
    def __init__(self, n=1_000, fs=16_000, fv=1_000, n_mics=100, name='db', betas=[0.92]*6, order=-1):
        """
        Args:
            n (int): Number of rooms to generate. 
            fs (int): Sampling frequency.
            fv (int): Frequency variable. 
            n_mics (int): Number of receiver microphones per room. 
            name (str): Name of the database file. 
            betas (list): Absorbtion coefficients.
            order (int): Maximum reflection order. Defaults to -1, which considers all.
        """
        self.n = n
        self.fs = fs
        self.fv = fv
        self.n_mics = n_mics
        self.filename = f"{name}-{str(n)}.npy"
        self.betas = betas
        self.order = order


    def __call__(self):
        """
        Checks to see if the database exists. 
        If it does, load it. If not, generate it.
        """
        if not os.path.isfile(self.filename):
            self.generate_database()
            self.save_file()
        return self.load_file()


    def normalize_db(self, db):
        """
        Normalize the database RIR values with a given range. 
        """
        def normalize(arr, t_min, t_max, x_min, x_max):
            """
            Normalize an array to a given range. 

            Args: 
                arr (np.array): Array to be normalized.
                t_min (float): Minimum value of the target range.
                t_max (float): Maximum value of the target range.
                x_min (float): Minimum value of the input range.
                x_max (float): Maximum value of the input range.

            Returns: 
                np.array: Normalized array.
            """
            norm_arr = []
            diff = t_max - t_min
            diff_arr = x_max - x_min   
            for i in arr:
                temp = (((i - x_min)*diff)/diff_arr) + t_min
                norm_arr.append(temp)
            return norm_arr

        # Find global min and max. 
        x_min = float('inf')
        x_max = -float('inf')
        for i in db:
            X,y = i 
            x_min = min([x_min, min(y)])
            x_max = max([x_max, max(y)])

        dbs = []
        for i in db: 
            X,y = i 
            y = normalize(y, 0, 1, x_min, x_max)
            dbs.append((X,y))

        return dbs

    def generate_database(self):
        """
        This function generates the database. 
        """
        instances = []
        points = 2048 
        
        for _ in tqdm(range(self.n)):
            room = [3,3,3] 
            # room = np.random.uniform(3,4,(3))
            # source = [1,1,1] 
            source = np.random.uniform(0,1,(3))
            r = 0.5
            center = [2,2,2] 
            center = np.random.uniform(1 + r + 0.5, 3 - r, (3)) # 0.5 is safey buffer between source and reciever.
            receivers = sample_random_receiver_locations(self.n_mics, r, center)
            features = [np.concatenate((room, source,receiver)) for receiver in receivers ] # Encode the room + source + receiver position.
            rir = frequency_rir(receivers, source, room, self.betas, points, self.fs, self.fv, order=self.order)
            rir = [r.real for r in rir] # Extract the real component. 
            instance = (features, rir)
            instances.append(instance)

        # self.db = self.normalize_db(instances)
        self.db = instances


    def save_file(self):   
        """
        Save the datbase as a numpy file.
        """  
        np.save(self.filename, self.db, allow_pickle=True)


    def load_file(self):
        """
        Loads the database from a numpy file.
        Splits the database into train and validation sets.
        """
        loaded_file = np.load(self.filename, allow_pickle=True, encoding='bytes')
        split = skorch.dataset.CVSplit(0.5) 
        train, validation = split(loaded_file)
        train = loaded_file[train.indices]
        validation = loaded_file[validation.indices]
        return train, validation    