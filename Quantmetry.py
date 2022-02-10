


import numpy as np
import pandas as pd 
import seaborn as sns 
from dython import nominal
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
from scipy.stats import norm, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# Class PreProcessing 
# Pour le prétraitement des données 
class PreProcessing : 

    def __init__(self, path : str) -> None:
        self.path = path
        self.df = pd.read_csv(self.path, index_col = 0).iloc[:, 1:]

    # Méthode nous permet de retourner dataframe
    def get_df(self):
        return self.df

    # Méthode nous permet de retourner la taille dataframe
    def get_shape(self):
        return self.df.shape

    # Méthode nous permet de recupérer des infos sur dataframe ( Type des variables ...)
    def get_info(self):
        return self.df.info()


    # Méthode nous permet d'avoir une idée sur les variables pertinentes
    def feature_selection(self):
        columns = ['cheveux', 'age', 'exp', 'salaire', 'sexe', 'diplome', 'specialite', 'note', 'dispo']
        aux_df = self.df.copy()
        # print(aux_df)
        aux_df.dropna(subset = aux_df.columns, inplace=True)
        # print(aux_df.shape)
        X = aux_df.iloc[: , 1:-1]
        y =  aux_df.iloc[: , -1]

        # Ordinal encoding 
        oe = OrdinalEncoder()
        oe.fit(X)
        X = oe.transform(X)

        model = ExtraTreesClassifier()
        model.fit(X,y)

        plt.figure(figsize=(12, 5), dpi=80)
        plt.xlabel("Mean decrease in impurity")
        feat_importances = pd.Series(model.feature_importances_, index= columns)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.show()
        self.reset_df()

    # Pour recupérer les indices des variables après le filtrage
    def get_columns_indices(self, num_col, cat_col):
        df = self.df.iloc[:,:-1]
        cols = df.columns
        num_indices = []
        col_indices = []

        num_cols = list(df._get_numeric_data().columns)
        cat_cols = list(set(cols) - set(num_cols))
        # print(num_cols)
        for el in list(num_col):
            num_cols.remove(el)

        for el in cat_col:
            cat_cols.remove(el)

        if 'date' in cat_cols:
            cat_cols.remove('date')

        for el in num_cols:
            num_indices.append(df.columns.get_loc(el))

        for el in cat_cols:
            col_indices.append(df.columns.get_loc(el))

        return col_indices, num_indices


    def get_target_X(self):
        return self.df.iloc[:,:-1].values, self.df.iloc[:,-1].values

    # Traitement des données manquantes : pour les 2 types de variables 
    def pre_process_data(self, num_col, cat_col):
        X, _ = self.get_target_X()
        cat_cols, num_cols = self.get_columns_indices(num_col, cat_col)

        # Categorical Processing 
        X_cat = np.copy(X[:, cat_cols])
        for col_id in range(len(cat_cols)):
            unique_val, val_idx = np.unique(X_cat[:, col_id].astype(str), return_inverse=True)
            X_cat[:, col_id] = val_idx
        imp_cat = SimpleImputer(missing_values=0, strategy='most_frequent')
        X_cat[:, :] = imp_cat.fit_transform(X_cat[:, :])

        # Numerical Processing 
        X_num = np.copy(X[:,num_cols])
        X_num = X_num.astype(float)
        imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_num = imp_num.fit_transform(X_num)

        return X_cat, X_num

    
    # Retourne les données après le prétraitement
    def get_preProcessed_data(self, num_col, cat_col):
        X_cat, X_num = self.pre_process_data(num_col, cat_col)
        X_cat_bin = OneHotEncoder().fit_transform(X_cat).toarray()
        X = np.concatenate((X_cat_bin, X_num), axis = 1)
        _, y = self.get_target_X()

        return X, y

    # Méthode qui permet d'appliqer OverSampling 
    def upSampling_df(self):
        df_majority = self.df[(self.df['embauche']==0)] 
        df_minority = self.df[(self.df['embauche']==1)] 
        # upsample minority class
        df_minority_upsampled = resample(df_minority, 
                                        replace=True,    # sample with replacement
                                        n_samples= 17708, # to match majority class
                                        random_state=42)  # reproducible results
        # Combine majority class with upsampled minority class
        self.df = pd.concat([df_minority_upsampled, df_majority])

    
    def reset_df(self):
        self.df = pd.read_csv(self.path, index_col = 0).iloc[:, 1:]



# Class Stats 
class Stats():
    def __init__(self, data) -> None:
        self.df = data

    def plot_target_hist(self):
        target = self.df.get_df().iloc[:,-1].values
        plt.figure(figsize= (6, 6))
        plt.hist(target, range = (0, 2), bins = 5, color = 'green',
            edgecolor = 'black')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.title('Histogramme of Target')

    def plot_correlation_heatmap(self):
        _, num_cols = self.df.get_columns_indices()
        correlations = self.df.get_df().iloc[: , num_cols].corr()
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                    square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
        plt.show()

    def plot_correlation(self):
        df_stats = self.df.get_df().copy()
        df_stats.dropna(subset = df_stats.columns, inplace=True)
        return nominal.associations(df_stats.iloc[: , 1:-1],figsize=(20,10),mark_columns=True)

    def plot_distribution(self, feature):
        if feature == 'exp':
            sns.distplot(self.df.get_df()[feature], fit = norm, color="y")
        else: 
            sns.histplot(self.df.get_df()[feature], color="green", 
                        label="100% Equities", kde=True, stat="density", linewidth=0.1)

    def calculate_dependency(self, var1, var2):
        df_stats = self.df.get_df().copy()
        df_stats.dropna(subset = df_stats.columns, inplace=True)
        cat_variables = ['cheveux','sexe', 'diplome', 'specialite','dispo']

        if (var1 in cat_variables) or (var2 in cat_variables):
            CrosstabResult=pd.crosstab(index= df_stats[var1], columns= df_stats[var2])
            ChiSqResult = chi2_contingency(CrosstabResult)
            return ChiSqResult[1]

        else : 
            corr, p_value = pearsonr(df_stats[var1], df_stats[var2])
            return corr, p_value