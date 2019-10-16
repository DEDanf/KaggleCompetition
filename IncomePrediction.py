import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from numpy import sqrt
from sklearn.tree import DecisionTreeClassifier



def meanNormalization(data):

    data = data.fillna(data.mean())
    normalized_df = (data - data.mean()) / data.std()

    return normalized_df

def processFullData(data):
    ##HEIGHT
    height = meanNormalization(pd.DataFrame(data, columns=['Body Height [cm]']))

    ##YEARS
    yearOfRecord = pd.DataFrame(data, columns=['Year of Record'])
    yearOfRecordNormalized = meanNormalization(yearOfRecord)

    ##AGE
    age = meanNormalization(pd.DataFrame(data, columns=['Age']))

    ##GENDER
    gender = pd.DataFrame(data, columns=['Gender'])
    gender = pd.get_dummies(gender)
    if 'Gender_0' in gender.columns:
        gender = gender.drop(["Gender_0"], axis=1)

    if 'Gender_unknown' in gender.columns:
        gender = gender.drop(["Gender_unknown"], axis=1)

    ##COUNTRY
    country = pd.DataFrame(data, columns=['Country'])
    country = pd.get_dummies(country)

    ##CITYSIZE
    citySize = pd.DataFrame(data, columns=['Size of City'])
    citySize = citySize.fillna(citySize.mean())

    ##PROFESSION
    profession = pd.DataFrame(data, columns=['Profession'])
    vc = profession.Profession.value_counts()
    newD = profession.replace(vc.index, vc)
    newD = meanNormalization(newD)


    ##UNIVERSITY_DEGREE
    university = pd.DataFrame(data, columns=['University Degree'])
    university = university.fillna(0)
    uni_types = (pd.get_dummies(university)).drop(["University Degree_0"], axis=1)

    ##GLASSES
    glasses = (pd.DataFrame(data, columns=['Wears Glasses'])).fillna(0)

    ##HAIR
    hairColour = pd.DataFrame(data, columns=['Hair Color'])
    hairColour = hairColour.fillna('Unknown')
    hairColour = pd.get_dummies(hairColour)

    if 'Hair Color_Unknown' in hairColour.columns:
        hairColour = hairColour.drop(["Hair Color_Unknown"], axis=1)

    if 'Hair Color_0' in hairColour.columns:
        hairColour = hairColour.drop(["Hair Color_0"], axis=1)

    ##COMBINING FEATURES

    new_data = pd.concat([yearOfRecordNormalized, height, age, glasses, gender, uni_types, hairColour,
                                citySize, country, newD], axis=1)

    return new_data


def processData(data,countriesList):

    ##TRAINING DATA INFO
    n = data.shape[0]
    features = data.shape[1] - 2

    ##HEIGHT
    height = meanNormalization(pd.DataFrame(data, columns=['Body Height [cm]']))

    ##YEARS
    yearOfRecord = pd.DataFrame(data, columns=['Year of Record'])
    yearOfRecordNormalized = meanNormalization(yearOfRecord)

    ##AGE
    age = meanNormalization(pd.DataFrame(data, columns=['Age']))

    ##GENDER
    gender = pd.DataFrame(data, columns=['Gender'])
    gender = pd.get_dummies(gender)
    if 'Gender_0' in gender.columns:
        gender = gender.drop(["Gender_0"], axis=1)

    if 'Gender_unknown' in gender.columns:
        gender = gender.drop(["Gender_unknown"], axis=1)

    ##COUNTRY
    country = pd.DataFrame(data, columns=['Country'])
    country = pd.get_dummies(country)
    zeros = np.zeros(n)
    for x in range(0, len(countriesList)):
        pd.DataFrame.insert(country, 0, countriesList[x], zeros)

    country = country.sort_index(axis=1)

    ##CITYSIZE
    citySize = pd.DataFrame(data, columns=['Size of City'])
    citySize = citySize.fillna(citySize.mean())

    ##PROFESSION
    profession = pd.DataFrame(data, columns=['Profession'])
    vc = profession.Profession.value_counts()
    newD = profession.replace(vc.index, vc)
    newD = meanNormalization(newD)


    ##UNIVERSITY_DEGREE
    university = pd.DataFrame(data, columns=['University Degree'])
    university = university.fillna(0)
    uni_types = (pd.get_dummies(university)).drop(["University Degree_0"], axis=1)

    ##GLASSES
    glasses = (pd.DataFrame(data, columns=['Wears Glasses'])).fillna(0)

    ##HAIR
    hairColour = pd.DataFrame(data, columns=['Hair Color'])
    hairColour = hairColour.fillna('Unknown')
    hairColour = pd.get_dummies(hairColour)

    if 'Hair Color_Unknown' in hairColour.columns:
        hairColour = hairColour.drop(["Hair Color_Unknown"], axis=1)

    if 'Hair Color_0' in hairColour.columns:
        hairColour = hairColour.drop(["Hair Color_0"], axis=1)

    ##COMBINING FEATURES
    if 'Income in EUR' in data.columns:
        y = pd.DataFrame(data, columns=['Income in EUR'])
        fixed_data = pd.concat([yearOfRecordNormalized, height, age, glasses, gender, uni_types, hairColour,
                            citySize, country, newD, y], axis=1)
    else:
        fixed_data = pd.concat([yearOfRecordNormalized, height, age, glasses, gender, uni_types, hairColour,
                                citySize, country, newD], axis=1)


    return fixed_data


def main():


    ##DATA INPUT

    training = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
    test = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
    concat1 = test.drop(['Income'], axis=1)
    concat2 = training.drop(['Income in EUR'], axis=1)
    index = concat2.shape[0]-1
    fullData = pd.concat([concat1, concat2], axis=0)
    fullData = fullData.drop(['Instance'], axis=1)
    processedFull = processFullData(fullData)
    index2 = processedFull.shape[0]-1

    #COUNTRY PROCESSING

    c1 = pd.get_dummies(pd.DataFrame(training, columns=['Country']))
    c2 = pd.get_dummies(pd.DataFrame(test, columns=['Country']))

    countriesList1 = np.setdiff1d(c1.columns, c2.columns)
    countriesList2 = np.setdiff1d(c2.columns, c1.columns)


    ##DATA PROCESSING

    processedTraining = processData(training,countriesList2)
    test = test.drop(['Income'], axis=1)
    processedTest = processData(test,countriesList1)


    ##Y values

    trainingTarget = pd.DataFrame(processedTraining, columns=['Income in EUR'])
    processedTraining = processedTraining.drop(['Income in EUR'], axis=1)



    ##MODEL SELECTION

    model = linear_model.Ridge(alpha=0.5)

    ##MODEL TRAINING

    model.fit(processedTraining,trainingTarget)


    ##MODEL TEST OUTPUT

    y_pred = model.predict(processedTest)
    pd.DataFrame(y_pred).to_csv('tcd ml 2019-20 income prediction submission file.csv', index_label='Instance', header=['Income'])
    print('Complete')

main()



