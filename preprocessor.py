import re
import pandas as pd

def preprocess(data):
    '''
    :param data:
    :return: a pandas dataframe
    '''
    regex_pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    messages = re.split(regex_pattern, data)[1:]
    dates = re.findall(regex_pattern, data)

    dataframe = pd.DataFrame({'user_message': messages, 'message_date': dates})
    try:
        dataframe['message_date'] = pd.to_datetime(dataframe['message_date'], format='%d/%m/%y, %H:%M - ')
    except:
        dataframe['message_date'] = pd.to_datetime(dataframe['message_date'], format='%m/%d/%y, %H:%M - ')
    dataframe.rename(columns={'message_date': 'date'}, inplace=True)

    users, messages = [], []
    for message in dataframe['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])                       # Username
            messages.append(" ".join(entry[2:]))         # Message
        else:
            users.append('group_notification')
            messages.append(entry[0])

    dataframe['user'] = users
    dataframe['message'] = messages

    dataframe.drop(columns=['user_message'], inplace=True)
    dataframe['only_date'] = dataframe['date'].dt.date
    dataframe['year'] = dataframe['date'].dt.year
    dataframe['month_num'] = dataframe['date'].dt.month
    dataframe['month'] = dataframe['date'].dt.month_name()
    dataframe['day'] = dataframe['date'].dt.day
    dataframe['day_name'] = dataframe['date'].dt.day_name()
    dataframe['hour'] = dataframe['date'].dt.hour
    dataframe['minute'] = dataframe['date'].dt.minute
    dataframe = dataframe[dataframe['user'] != 'group_notification']

    return dataframe
