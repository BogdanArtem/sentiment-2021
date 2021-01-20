import os
from sklearn import preprocessing
import sys
import numpy as np
import pandas as pd
import pickle
from azureml.dataprep import package
sys.path.append(".")
sys.path.append("..")

removedWordsList = (['xxxxx1'])


def removeNonUkr(text, ukrWords):
    global removedWordsList
    wordList = text.split()
    if len(wordList) == 0:
        return " "
    y = np.array(wordList)
    x = np.array(ukrWords)
    index = np.arange(len(ukrWords))
    sorted_index = np.searchsorted(x, y)
    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] != y
    maskedArr = np.ma.array(yindex, mask=mask).compressed()
    result = x[maskedArr]
    text = np.array2string(result)\
        .replace("\'", "")\
        .replace("[", "")\
        .replace("]", "")\
        .replace("\n", "")\
        .replace("\r", "")

    # Logging removed words
    removedWords = set(wordList)-set(result)
    removedWordsList += set(list(removedWords))-set(removedWordsList)
    return text


def encryptSingleColumn(data):
    le = preprocessing.LabelEncoder()
    le.fit(data)
    return le.transform(data)


def encryptColumnsCollection(data, columnsToEncrypt):
    for column in columnsToEncrypt:
        data[column] = encryptSingleColumn(data[column])
    return data


def removeString(data, regex):
    return data.str.lower().str.replace(regex.lower(), ' ')


def cleanDataset(dataset, columnsToClean, regexList):
    for column in columnsToClean:
        for regex in regexList:
            dataset[column] = removeString(dataset[column], regex)
    return dataset


def getRegexList():
    regexList = []
    regexList += ['From:(.*)\r\n']  
    regexList += ['Sent:(.*)\r\n']  
    regexList += ['Received:(.*)\r\n']  
    regexList += ['To:(.*)\r\n']  
    regexList += ['CC:(.*)\r\n']  
    regexList += ['The information(.*)infection']  
    regexList += ['Endava Limited is a company(.*)or omissions']  
    regexList += ['The information in this email is confidential and may be legally(.*)interference if you are not the intended recipient']  # footer
    regexList += ['\[cid:(.*)]']  
    regexList += ['https?:[^\]\n\r]+']  
    regexList += ['Subject:']

    regexList += ['^[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$']
    regexList += ['[\w\d\-\_\.]+ @ [\w\d\-\_\.]+']
    regexList += ['Subject:']
    regexList += ['[^a-zA-Z]']

    return regexList


if __name__ == '__main__':
    dfIncidents = package.run('IncidentsCleaned.dprep', dataflow_idx=0)
    dfRequests = package.run('RequestsCleaned.dprep', dataflow_idx=0)
    columnsOrder = [
        'title', 'body', 'ticket_type', 'category',
        'sub_category1', 'sub_category2', 'business_service',
        'urgency', 'impact'
    ]
    dfIncidents = dfIncidents[columnsOrder]
    dfRequests = dfRequests[columnsOrder]

    dfTickets = dfRequests.append(
        dfIncidents,
        ignore_index=True)  

    columnsToDropDuplicates = ['body']
    dfTickets = dfTickets.drop_duplicates(columnsToDropDuplicates)

    columnsToClean = ['body', 'title']

    cleanDataset(dfTickets, columnsToClean, getRegexList())

    dfWordsEn = package.run('WordsEn.dprep', dataflow_idx=0)
    dfFirstNames = package.run('FirstNames.dprep', dataflow_idx=0)
    dfBlackListWords = package.run('WordsBlacklist.dprep', dataflow_idx=0)

    dfWordsEn['Line'] = dfWordsEn['Line'].str.lower()
    dfFirstNames['Line'] = dfFirstNames['Line'].str.lower()
    dfBlackListWords['Line'] = dfBlackListWords['Line'].str.lower()

    dfWords = dfWordsEn.merge(
        dfFirstNames.drop_duplicates(),
        on=['Line'], how='left', indicator=True)
    dfWords = dfWords.loc[dfWords['_merge'] == 'left_only']

    dfWords = dfWords.drop("_merge", axis=1)  # Drop merge indicator column

    dfWords = dfWords.merge(
        dfBlackListWords.drop_duplicates(),
        on=['Line'], how='left', indicator=True
    )
    dfWords = dfWords.loc[dfWords['_merge'] == 'left_only']
    print("Shape after removing blacklisted\
        words from english words dataset: "+str(dfWords.shape))

    dfTickets['body'] = dfTickets['body'].apply(
        lambda emailBody: removeNonEnglish(emailBody, dfWords['Line']))
    dfTickets['title'] = dfTickets['title'].apply(
        lambda emailBody: removeNonEnglish(emailBody, dfWords['Line']))

    print("Before removing empty: " + str(dfTickets.shape))
    dfTickets = dfTickets[dfTickets.body != " "]
    dfTickets = dfTickets[dfTickets.body != ""]
    dfTickets = dfTickets[~dfTickets.body.isnull()]
    print("After removing empty: " + str(dfTickets.shape))

    columnsToDropDuplicates = ['body']
    dfTickets = dfTickets.drop_duplicates(columnsToDropDuplicates)
    print(dfTickets.shape)

    # Save 
    dfTickets.to_csv('all_tickets.csv', index=False, index_label=False)
    sortedRemovedWordsList = np.sort(removedWordsList)
    dfx = pd.DataFrame(sortedRemovedWordsList)
    dfx.to_csv("removed_words.csv", index=False, index_label=False)
