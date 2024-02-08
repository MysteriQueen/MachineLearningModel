from argparse import ArgumentParser
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def RetrieveDataFrame(dfHeartDiseasesArg):
    # Read heart disease dataset and store into a dataframe variable
    dfHeartDiseases = pd.read_csv(dfHeartDiseasesArg, header=None, sep=',')
    # Print dataframe to console
    print("CHECK || RetrieveDataFrame() || Heart Disease DataFrame:")
    print(dfHeartDiseases)

    # Add column names to the dataframe
    dfHeartDiseases.columns = ['Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure', 'Cholesterol',
                               'Fasting Blood Sugar', 'ECG', 'Max Heart Rate', 'Exercise Angina', 'ST Depression',
                               'Slope', 'Major Vessels Blocked', 'Thallium Test', 'Diagnosis']
    # Print column names to console
    print("CHECK || RetrieveDataFrame() || Column names are:\n" + str(dfHeartDiseases.columns))
    # Print information
    dfHeartDiseases.info()
    # Replace 2s, 3s and 4s with 1 in the Target Variable column
    dfHeartDiseases["Diagnosis"] = dfHeartDiseases["Diagnosis"].replace([2, 3, 4], 1)

    # Export dataframe to an Excel File
    dfHeartDiseases.to_excel('HeartDiseasesDataset.xlsx')
    # Print message to console
    print("CHECK || RetrieveDataFrame() || Dataframe is written to Excel File successfully.")

    return dfHeartDiseases


def PrepareData(dfHeartDiseases):
    # Print shape value to the console
    print("\nCHECK || PrepareData() || ORIGINAL Shape: " + str(dfHeartDiseases.shape))
    # Check distribution of features and print to console
    for oColumn in dfHeartDiseases:
        print("\nCHECK || PrepareData() || Distribution BEFORE: " + oColumn + "\n" + str(dfHeartDiseases[oColumn].value_counts()))

    # DATA PREPROCESSING: Invoke function - Clean the dataframe
    dfPreprocessed = PreprocessData(dfHeartDiseases)

    # Check distribution of features and print to console
    for oColumn in dfPreprocessed:
        print("\nCHECK || PrepareData() || Distribution AFTER: " + oColumn + "\n" + str(dfPreprocessed[oColumn].value_counts()))

    return dfPreprocessed


def PreprocessData(dfHeartDiseases):
    # Remove missing values encoded as '?'
    dfHeartDiseases = dfHeartDiseases[(dfHeartDiseases.astype(str) != '?').all(axis=1)]
    # Reset index
    dfHeartDiseases = dfHeartDiseases.reset_index(drop=True)
    # Print shape value to the console
    print("\nCHECK || PreprocessData() || Shape AFTER removing missing values: " + str(dfHeartDiseases.shape))

    for oColumn in dfHeartDiseases:
        # Change non numerical values to numerical
        dfHeartDiseases[oColumn] = pd.to_numeric(dfHeartDiseases[oColumn])

    # Invoke function, passing the dataframe and returning the dataframe without outliers
    dfHeartDiseasesClean = RemoveOutliers(dfHeartDiseases)
    # Reset index
    dfHeartDiseasesClean = dfHeartDiseasesClean.reset_index(drop=True)
    # Export CLEAN dataframe to another Excel File
    dfHeartDiseasesClean.to_excel('HeartDiseasesDatasetClean.xlsx')
    # Print message to console
    print("CHECK || PreprocessData() || CLEAN Dataframe is written to Excel File successfully.")

    # Scale the data
    oStandardScaler = StandardScaler()
    columns_to_scale = ['Age', 'Resting Blood Pressure', 'Cholesterol', 'Max Heart Rate', 'ST Depression']
    dfHeartDiseasesClean[columns_to_scale] = oStandardScaler.fit_transform(dfHeartDiseasesClean[columns_to_scale])
    # Write to an Excel file
    dfHeartDiseasesClean.to_excel('HeartDiseasesDatasetScale.xlsx')

    return dfHeartDiseasesClean


def RemoveOutliers(dfHeartDiseases):
    # Export statistics of the dataframe to an Excel file
    dfHeartDiseases.describe().to_excel('Statistics.xlsx')
    # Print shape value to the console
    print("\nCHECK || RemoveOutliers() || Shape BEFORE: " + str(dfHeartDiseases.shape))

    # Retrieve Q1 and Q3
    fQ1 = dfHeartDiseases.quantile(0.25)
    fQ3 = dfHeartDiseases.quantile(0.75)
    # Calculate IQR
    fIQR = fQ3 - fQ1
    # Print IQR to console
    print("\nCHECK || RemoveOutliers() || IQR:\n" + str(fIQR))

    # Detect outliers by storing a boolean, produced by dfHeartDiseases >= (fQ3 + 1.5 * fIQR, into a dataframe
    dfUpperBound = dfHeartDiseases >= (fQ3 + 1.5 * fIQR)
    dfLowerBound = dfHeartDiseases <= (fQ1 - 1.5 * fIQR)
    # Print bound dataframes to console
    print("\nCHECK || RemoveOutliers() || \nUpper Bound:\n" + str(dfUpperBound) + "\nLower Bound:\n" + str(dfLowerBound))
    # Export bound dataframes to an Excel file
    dfUpperBound.to_excel('UpperBoundOutliers.xlsx')
    dfLowerBound.to_excel('LowerBoundOutliers.xlsx')

    # Remove outliers by row, using the IQR score
    dfHeartDiseasesClean = dfHeartDiseases[~((dfHeartDiseases > (fQ3 + 1.5 * fIQR)) | (dfHeartDiseases < (fQ1 - 1.5 * fIQR))).any(axis=1)]
    # Print shape of the dataframe to console
    print("\nCHECK || RemoveOutliers() || Shape AFTER: " + str(dfHeartDiseasesClean.shape))

    return dfHeartDiseasesClean


def SupervisedLearning(dfHeartDiseasesPrepared):
    # Store the target feature column into another dataframe
    dfTargetFeature = dfHeartDiseasesPrepared["Diagnosis"]
    # Remove the target feature column from the dataframe
    dfFeatures = dfHeartDiseasesPrepared.drop(["Diagnosis"], axis=1)
    # Export both dataframes to Excel files
    dfTargetFeature.to_excel('TargetFeature.xlsx')
    dfFeatures.to_excel('Features.xlsx')

    # Split dataset into training and testing datasets at 75% training data and 25% test data
    dfFeaturesTrain, dfFeaturesTest, dfTargetFeatureTrain, dfTargetFeatureTest = train_test_split(dfFeatures, dfTargetFeature, test_size=0.25, random_state=0)
    # Print on console
    print("Features Train: " + str(dfFeaturesTrain.shape))
    print("Features Test: " + str(dfFeaturesTest.shape))
    print("Target Feature Train: " + str(dfTargetFeatureTrain.shape))
    print("Target Feature Test: " + str(dfTargetFeatureTest.shape))

    # Model Training
    oModel = LogisticRegression()
    # Train the LogisticRegression Model with Training Data
    oModel.fit(dfFeaturesTrain.values, dfTargetFeatureTrain.values)

    # Model Evaluation - Training Data
    dfFeaturesTrainPrediction = oModel.predict(dfFeaturesTrain.values)
    fTrainingDataAccuracy = accuracy_score(dfFeaturesTrainPrediction, dfTargetFeatureTrain.values)
    print("Training Data Accuracy: " + str(fTrainingDataAccuracy))

    # Model Evaluation - Test Data
    dfFeaturesTestPrediction = oModel.predict(dfFeaturesTest.values)
    fTestDataAccuracy = accuracy_score(dfFeaturesTestPrediction, dfTargetFeatureTest.values)
    print("Test Data Accuracy: " + str(fTestDataAccuracy))

    # Building a Predictive System
    # Input data
    tpInputData = (58.0,1.0,2.0,120.0,284.0,0.0,2.0,160.0,0.0,1.8,2.0,0.0,3.0)
    # Change input data tuple to np array
    arInputData = np.asarray(tpInputData)
    # Reshape the np array to predict for only 1 instance
    arInputDataReshape = arInputData.reshape(1, -1)
    lPrediction = oModel.predict(arInputDataReshape)
    # Print to console
    print(lPrediction)

    if lPrediction[0] == 0:
        print("The person does not have a high risk of getting a heart attack")
    else:
        print("The person does have a high risk of getting a heart attack")


def Main():
    # Initiate argument parser for console
    oParser = ArgumentParser(description="Artificial Intelligence Coursework")
    # Add argument for the dataset file
    oParser.add_argument("Dataset", help="File containing dataset.", default="processed.cleveland.data", nargs="?")
    # Store the argument into a variable
    oArgs = oParser.parse_args()
    # Print to console
    print("CHECK || Main() || Arguments: " + str(oArgs))

    # Invoke function, passing the dataset argument and returning the dataframe
    dfHeartDiseases = RetrieveDataFrame(oArgs.Dataset)
    # Invoke function, passing the dataframe and retuning the prepared version of the dataframe
    dfHeartDiseasesPrepared = PrepareData(dfHeartDiseases)
    # Invoke function, passing the preprocessed dataframe
    SupervisedLearning(dfHeartDiseasesPrepared)


# Invoke Main function
if __name__ == '__main__':
    Main()
