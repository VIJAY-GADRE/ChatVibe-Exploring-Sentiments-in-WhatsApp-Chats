# Importing modules
import matplotlib
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import streamlit as st

import helper
import preprocessor

matplotlib.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)

# Setting the title of our app
st.sidebar.title("ChatVibe: Exploring Sentiments in WhatsApp Chats")

# VADER is a sentiment analysis tool that utilizes a lexicon and rule-based approach, specifically designed to capture and interpret sentiments.
nltk.download('vader_lexicon')

# Creating a File "UPLOAD" button
uploaded_file = st.sidebar.file_uploader("Upload a file")

# Creating the Main Header
st. markdown("<h1 style='text-align: center; color: grey;'>ChatVibe: Exploring Sentiments in WhatsApp Chats</h1>", unsafe_allow_html=True)

if uploaded_file is not None:
    
    # Decoding the Byte form of the data
    bytes_data = uploaded_file.getvalue()
    d = bytes_data.decode("utf-8")
    
    # Preprocessing the data
    data = preprocessor.preprocess(d)
    
    # SentimentIntensityAnalyzer class is being imported from the "nltk.sentiment.vader" module.
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sentiments = SentimentIntensityAnalyzer()
    
    # Creating distinct columns -> (Positive / Neutral / Negative)
    data["po"] = [sentiments.polarity_scores(i)["pos"] for i in data["msg"]] # Positive
    data["ne"] = [sentiments.polarity_scores(i)["neg"] for i in data["msg"]] # Negative
    data["nu"] = [sentiments.polarity_scores(i)["neu"] for i in data["msg"]] # Neutral
    
    # indentify true sentiment per row in msg column
    def sentiment(d):
        '''
        Function to identify TRUE sentiments for each row in the msg column
        :param d: a dataset
        :return: an integer
                  1 -> Positive
                  0 -> Neutral
                 -1 -> Negative
        '''
        if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
            return 1
        if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
            return -1
        if d["nu"] >= d["po"] and d["nu"] >= d["ne"]:
            return 0

    # Finding the sentiments of each msg and storing it in new column, 'value'
    data['value'] = data.apply(lambda row: sentiment(row), axis=1)
    user_list = data['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")
    
    # Selectbox
    selected_user = st.sidebar.selectbox("Display the analysis in relation to", user_list)
    
    if st.sidebar.button("Display Analysis"):
        # Code to plot MONTHLY ACTIVITY MAP
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>MONTHLY ACTIVITY MAP (Positive)</h3>", unsafe_allow_html=True)
            
            busy_month = helper.month_activity_map(selected_user, data,1)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>MONTHLY ACTIVITY MAP (Neutral)</h3>", unsafe_allow_html=True)
            
            busy_month = helper.month_activity_map(selected_user, data, 0)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>MONTHLY ACTIVITY MAP (Negative)</h3>",unsafe_allow_html=True)
            
            busy_month = helper.month_activity_map(selected_user, data, -1)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Code to plot DAILY ACTIVITY MAP
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>DAILY ACTIVITY MAP (Positive)</h3>",unsafe_allow_html=True)
            
            busy_day = helper.week_activity_map(selected_user, data,1)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>DAILY ACTIVITY MAP (Neutral)</h3>",unsafe_allow_html=True)
            
            busy_day = helper.week_activity_map(selected_user, data, 0)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>DAILY ACTIVITY MAP (Negative)</h3>",unsafe_allow_html=True)
            
            busy_day = helper.week_activity_map(selected_user, data, -1)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Code to plot WEEKLY ACTIVITY MAP
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>WEEKLY ACTIVITY MAP (Positive)</h3>",unsafe_allow_html=True)
                
                user_heatmap = helper.activity_heatmap(selected_user, data, 1)
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')

        with col2:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>WEEKLY ACTIVITY MAP (Neutral)</h3>",unsafe_allow_html=True)
                
                user_heatmap = helper.activity_heatmap(selected_user, data, 0)
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')
        with col3:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>WEEKLY ACTIVITY MAP (Negative)</h3>",unsafe_allow_html=True)
                
                user_heatmap = helper.activity_heatmap(selected_user, data, -1)
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')

        # Code to plot DAILY TIMELINE
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>DAILY TIMELINE (Positive)</h3>", unsafe_allow_html=True)
            
            daily_timeline = helper.daily_timeline(selected_user, data, 1)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['msg'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>DAILY TIMELINE (Neutral)</h3>", unsafe_allow_html=True)
            
            daily_timeline = helper.daily_timeline(selected_user, data, 0)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['msg'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>DAILY TIMELINE (Negative)</h3>",unsafe_allow_html=True)
            
            daily_timeline = helper.daily_timeline(selected_user, data, -1)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['msg'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Code to plot MONTHLY TIMELINE
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>MONTHLY TIMELINE (Positive)</h3>", unsafe_allow_html=True)
            
            timeline = helper.monthly_timeline(selected_user, data,1)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['msg'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>MONTHLY TIMELINE (Neutral)</h3>", unsafe_allow_html=True)
            
            timeline = helper.monthly_timeline(selected_user, data,0)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['msg'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>MONTHLY TIMELINE (Negative)</h3>", unsafe_allow_html=True)
            
            timeline = helper.monthly_timeline(selected_user, data,-1)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['msg'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Code to plot MOST PERCENTAGE CONTRIBUTED
        if selected_user == 'Overall':
            col1,col2,col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;'>MOST POSITIVE CONTRIBUTION</h3>",unsafe_allow_html=True)
                x = helper.percentage(data, 1)
                st.dataframe(x)

            with col2:
                st.markdown("<h3 style='text-align: center; color: black;'>MOST NEUTRAL CONTRIBUTION</h3>",unsafe_allow_html=True)
                y = helper.percentage(data, 0)
                st.dataframe(y)

            with col3:
                st.markdown("<h3 style='text-align: center; color: black;'>MOST NEGATIVE CONTRIBUTION</h3>",unsafe_allow_html=True)
                z = helper.percentage(data, -1)
                st.dataframe(z)

        # Code for MOST POSITIVE, NEUTRAL, AND NEGATIVE USER...
        if selected_user == 'Overall':

            # Getting names per sentiment
            x = data['user'][data['value'] == 1].value_counts().head(10)
            y = data['user'][data['value'] == -1].value_counts().head(10)
            z = data['user'][data['value'] == 0].value_counts().head(10)

            col1,col2,col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;'>MOST POSITIVE USERS</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.markdown("<h3 style='text-align: center; color: black;'>MOST NEUTRAL USERS</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.bar(z.index, z.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col3:
                st.markdown("<h3 style='text-align: center; color: black;'>MOST NEGATIVE USERS</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.bar(y.index, y.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

        # Code to plot a WORDCLOUD
        col1,col2,col3 = st.columns(3)
        with col1:
            try:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>POSITIVE WORDCLOUD</h3>",unsafe_allow_html=True)
                df_wc = helper.create_wordcloud(selected_user, data,1)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                st.image('error.webp')

        with col2:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>NEUTRAL WORDCLOUD</h3>",unsafe_allow_html=True)
                df_wc = helper.create_wordcloud(selected_user, data,0)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                st.image('error.webp')

        with col3:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>NEGATIVE WORDCLOUD</h3>",unsafe_allow_html=True)
                df_wc = helper.create_wordcloud(selected_user, data,-1)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)

            except:
                st.image('error.webp')

        # Code for MOST POSITIVE COMMON WORDS
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                most_common_df = helper.most_common_words(selected_user, data,1)
                st.markdown("<h3 style='text-align: center; color: black;'>POSITIVE COMMON WORDS</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1],color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                st.image('error.webp')

        with col2:
            try:
                most_common_df = helper.most_common_words(selected_user, data,0)
                st.markdown("<h3 style='text-align: center; color: black;'>POSITIVE NEUTRAL WORDS</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1],color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                st.image('error.webp')

        with col3:
            try:
                most_common_df = helper.most_common_words(selected_user, data,-1)
                st.markdown("<h3 style='text-align: center; color: black;'>NEGATIVE COMMON WORDS</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                st.image('error.webp')
