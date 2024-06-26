import numpy as np 
import pandas as pd
import os 

class feature_generation:
    def __init__(self, data):
        ##  input: data, np.array (n,2,180)
        ##  output: concat feature  (n,360)
        ##  data[:,0,:]:  blood oxygen  data[:,1,:]: heart rate

        self.feature_dict = {}
        self.feature_dict['bo'] = data[:,0,:]
        self.feature_dict['hr'] = data[:,1,:]
        self.bo = data[:,0,:]
        self.hr = data[:,1,:]
        self.N = data.shape[0]
        return None 

    def bo_hr_ratio(self):
        self.feature_dict['bo_hr_ratio'] = self.bo / self.hr 
        return None 

    def bo_hr_product(self):
        self.feature_dict['bo_hr_ratio'] = self.bo * self.hr
        return None
    
    def bo_hr_plus(self):
        self.feature_dict['bo_hr_plus'] = self.bo + self.hr
        return None
    
    def bo_hr_minus(self):
        self.feature_dict['bo_hr_minus'] = self.bo - self.hr
        return None

    def statistics_feature(self):
        self.feature_dict['bo_max'] = self.bo.max(axis = 1)
        self.feature_dict['bo_min'] = self.bo.min(axis = 1)
        self.feature_dict['bo_mean'] = self.bo.mean(axis = 1)
        self.feature_dict['bo_std'] = self.bo.std(axis = 1)
        self.feature_dict['bo_diff'] = self.bo.max(axis = 1) - self.bo.min(axis = 1)

        self.feature_dict['hr_max'] = self.hr.max(axis = 1)
        self.feature_dict['hr_min'] = self.hr.min(axis = 1)
        self.feature_dict['hr_mean'] = self.hr.mean(axis = 1)
        self.feature_dict['hr_std'] = self.hr.std(axis = 1)
        self.feature_dict['hr_diff'] = self.hr.max(axis = 1) - self.hr.min(axis = 1)
        return None
    

    def get_feature(self):
        self.bo_hr_minus()
        self.bo_hr_plus()
        self.bo_hr_product()
        self.bo_hr_ratio()
        self.statistics_feature()
        return None

    def feature_concat(self):
        feature_name = []
        X = np.empty(shape=(self.N,0))
        for key in self.feature_dict:
            sub_feature = self.feature_dict[key]
            try:
                for i in range(sub_feature.shape[1]):
                    feature_name.append(f'{key}_{i}')
                X = np.concatenate([X, sub_feature], axis = 1)
            except:
                feature_name.append(f'{key}')
                X = np.concatenate([X, sub_feature[:,np.newaxis]], axis = 1)
        self.X = X
        self.feature_name = feature_name
        return None