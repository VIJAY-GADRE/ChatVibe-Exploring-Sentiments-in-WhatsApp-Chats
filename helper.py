from collections import Counter

import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud

extract = URLExtract()
'''
-1 -> Negative
0 -> Neutral
1 -> Positive
'''

def week_activity_map(selected_user, data, sentiment):
    '''
    :param selected_user:
    :param data:
    :param sentiment:
    :return: an integer of Count of Messages per Day
    '''
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]
    data = data[data['value'] == sentiment]
    return data['day_name'].value_counts()

def month_activity_map(selected_user, data, sentiment):
    '''
    :param selected_user:
    :param data:
    :param sentiment:
    :return: an integer of Count of Messages per Month
    '''
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]
    data = data[data['value'] == sentiment]
    return data['month'].value_counts()

def activity_heatmap(selected_user, data, sentiment):
    '''
    :param selected_user:
    :param data:
    :param sentiment:
    :return: a heatmap of Count of Messages
    '''
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]
    data = data[data['value'] == sentiment]

    user_heatmap = data.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap

def daily_timeline(selected_user, data, sentiment):
    '''
    :param selected_user:
    :param data:
    :param sentiment:
    :return: a dataframe of Count of Messages per Date
    '''
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]
    data = data[data['value'] == sentiment]
    # count of message on a specific date
    daily_timeline = data.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

def monthly_timeline(selected_user, data, sentiment):
    '''
    :param selected_user:
    :param data:
    :param sentiment:
    :return: a dataframe of Count of Messages per (year + month number + month)
    '''
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]
    data = data[data['value'] == -sentiment]
    timeline = data.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

def percentage(data, sentiment):
    '''
    :param data:
    :param sentiment:
    :return: a dataframe
    '''
    data = round(
        (data['user'][data['value'] == sentiment].value_counts() / data[data['value'] == sentiment].shape[0]) * 100,
        2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return data

def create_wordcloud(selected_user, data, sentiment):
    '''
    :param selected_user:
    :param data:
    :param sentiment:
    :return: Return a wordcloud
    '''
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]

    temp = data[data['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    # Remove stop words -> "stop_hinglish.txt"
    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    temp['message'] = temp['message'][temp['value'] == sentiment]

    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, data, sentiment):
    '''
    :param selected_user:
    :param data:
    :param sentiment:
    :return: a dataframe containing most common words
    '''
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]
    temp = data[data['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    words = []
    for message in temp['message'][temp['value'] == sentiment]:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df
