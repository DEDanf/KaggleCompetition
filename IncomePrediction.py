import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


def meanNormalization(data):

    data = data.fillna(data.mean())
    normalized_df = (data - data.mean()) / data.std()

    ##for x in range(0,n):

    return normalized_df


def binarizing(data):

    filled=data.fillna(0)

    return filled


def main():


    training = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
    test = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')

    n = training.shape[0]
    features = training.shape[1] - 2

    target = pd.DataFrame(training, columns=['Income in EUR'])
    height = meanNormalization(pd.DataFrame(training, columns=['Body Height [cm]']))

    yearOfRecord = meanNormalization(pd.DataFrame(training, columns=['Year of Record']))

    gender = pd.DataFrame(training, columns=['Gender'])
    gender_fixed = (pd.get_dummies(gender)).drop(["Gender_0", "Gender_unknown"], axis=1)

    age = meanNormalization(pd.DataFrame(training, columns=['Age']))

    country = pd.DataFrame(training, columns=['Country'])
    num_country = pd.get_dummies(country)
    print(num_country)

    citySize = pd.DataFrame(training, columns=['Size of City'])

    profession = pd.DataFrame(training, columns=['Profession'])
    profession_types = pd.get_dummies(profession)
    ##print(profession_types)

    university = pd.DataFrame(training, columns=['University Degree'])
    university = university.fillna(0)
    uni_types = (pd.get_dummies(university)).drop(["University Degree_0"], axis=1)

    glasses = binarizing(pd.DataFrame(training, columns=['Wears Glasses']))

    hairColour = pd.DataFrame(training, columns=['Hair Color'])
    hairTypes = pd.get_dummies(hairColour)
    print(hairTypes)

    hor_stack=pd.concat([yearOfRecord,height,age,glasses,gender_fixed,uni_types],axis=1)

    polynomial_features = PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(hor_stack)

    model = linear_model.LinearRegression()
    model.fit(x_poly,target)
    y_pred = model.predict(x_poly)
    print(mean_squared_error(target,y_pred))



main()



