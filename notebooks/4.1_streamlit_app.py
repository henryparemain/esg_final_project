import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import plotly.express as px
import clean_pdf as cp
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk.corpus
from nltk.corpus import stopwords
import string
import re
import requests
import pdftotext
import io
import subprocess
import joblib
import numpy as np
import gensim 
from gensim import corpora
import pyLDAvis.gensim_models
from sklearn.preprocessing import MinMaxScaler

# Set page title
st.set_page_config(page_title="ESG Insight App", layout="wide")


# Add sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["2021 Insights", "Custom Insights"])






# -----PAGE 1------

# Add content based on page selection
if page == "2021 Insights":
    st.title("2021 Insights")
    st.write("This page contains insights from the 2021 ESG reports. Explore the most common terms by industry, compare company scores, and search key terms within a company's report!")
    df = pd.read_csv('./data/df_merge.csv')
    # df = df.drop(columns = ['Unnamed: 0'])
    df.set_index('ticker',inplace=True)


    my_words = df.iloc[:,6:]
    #drop noisy words
    # noisy_words = ['percent','client','associate','corporate responsibility','appendix','fiscal']
    # my_words = my_words.drop(columns = noisy_words)
    # #calculate totals for entire corpus
    # word_weights = my_words.values.sum(axis=0)

    st.write("")
    row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
        (0.1, 1, 0.1, 1, 0.1))
    
    sector_list = list(df.sector.unique())
    with row3_1:
        #plot bar chart 1
        selected_ind = st.selectbox("Select an Industry", options= sector_list)
        df = df.drop(columns = ['fiscal year','fiscal','client','percent','percent','client','associate','corporate responsibility','appendix','fiscal'])
        df_vectorized = df.iloc[:,4:]
        df_vectorized = df_vectorized.groupby('sector').sum()
        df_vectorized_array = df_vectorized.values
        
        for j in range(9):
          values_di = {}
          for i in range(len(df_vectorized_array)):
              words = []
              values = []
              top_ten_indices = np.argsort(-df_vectorized_array[i])[:10]
              for idx in top_ten_indices:
                  words.append(df_vectorized.columns[idx])
                  values.append(df_vectorized_array[i][idx])
              values_di[df_vectorized.index[i]] = dict(zip(words, values))
        
        df_clean_ind = pd.DataFrame.from_dict(values_di[selected_ind], orient='index', columns=['count'])

        
        # words_df = pd.DataFrame({"token": my_words.columns, 
        #                         "weights": word_weights})
        
        # words_df = words_df.sort_values(by="weights", ascending=False).head(20)

        fig = px.bar(
            df_clean_ind,
            x=df_clean_ind.index,
            y="count",
            title= f'Top 10 Most Frequent Terms - {selected_ind}',
            color_discrete_sequence=["#5BAA67"],
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)


    with row3_2:
        # Create a dropdown menu with the ticker symbols
        selected_cat = st.selectbox("Select a Ticker", options=['Environmental Scores','Social Scores','Governance Scores'])
        df_top_3_E = df.sort_values(['E'],ascending=False)['E'][:10]
        df_top_3_S = df.sort_values(['S'],ascending=False)['S'][:10]
        df_top_3_G = df.sort_values(['G'],ascending=False)['G'][:10]

        if selected_cat == 'Environmental Scores':
            fig = px.bar(
            df_top_3_E,
            x= df_top_3_E.index,
            y='E',
            title= f'Top 10 Highest Environmental Scores',
            color_discrete_sequence=["#5BAA67"],
            )
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)


        if selected_cat == 'Social Scores':
            fig = px.bar(
            df_top_3_S,
            x= df_top_3_S.index,
            y='S',
            title= f'Top 10 Highest Social Scores',
            color_discrete_sequence=["#5BAA67"],
            )
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)


        if selected_cat == 'Governance Scores':
            fig = px.bar(
            df_top_3_G,
            x= df_top_3_G.index,
            y='G',
            title= f'Top 10 Highest Governance Scores',
            color_discrete_sequence=["#5BAA67"],
            )
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)


    st.subheader('Explore a particular company below')
    # Create a dropdown menu with the ticker symbols
    selected_ticker = st.selectbox("Select a Ticker", options=df.index)

    # Display the score for the selected ticker
    e_score = df.loc[df.index == selected_ticker, "E"].iloc[0]
    st.write(f"Environmental Score: {e_score}")
    s_score = df.loc[df.index == selected_ticker, "S"].iloc[0]
    st.write(f"Social Score: {s_score}")
    g_score = df.loc[df.index == selected_ticker, "G"].iloc[0]
    st.write(f"Governance Score: {g_score}")

    # plotting a second bar chart that links to the user selected company
    #selecting only the columns with TFIDF values
    df_vals = df.iloc[:,5:]
    #selecting the company based on the user selection
    word_weights_ticker = df_vals.loc[selected_ticker,:]
    #form temporary dataframe
    words_df = pd.DataFrame({"token": df_vals.columns, 
                             "weights": word_weights_ticker})
    #sort tokens and select top 20
    words_df = words_df.sort_values(by="weights", ascending=False).head(20)

    fig = px.bar(
        words_df,
        x="token",
        y="weights",
        title= f'Top 20 Most Frequent Terms for {selected_ticker}',
        color_discrete_sequence=["#5BAA67"],
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)













# -----PAGE 2------


elif page == "Custom Insights":
    st.title("Custom Insights")
    st.write("This page contains customized insights. Upload a PDF to gain an ESG score; explore the key paragraphs relating to ESG; and explore the key topics within the report!")

    # Allow the user to upload a PDF file
    uploaded_file = st.file_uploader('Upload a PDF file', type='pdf')

    # If a file was uploaded
    if uploaded_file is not None:
        with st.spinner("Calculating scores, please wait..."):
            # Read the contents of the file
            pdf_data = io.BytesIO(uploaded_file.read())
            # Use pdftotext to extract the text from the PDF file
            result = subprocess.run(["pdftotext", "-", "-"], input=pdf_data.getvalue(), capture_output=True)

            # Convert the result from bytes to a string and display it to the user
            text = result.stdout.decode("utf-8")
            df = cp.clean_pdf(text)
            # Display a download link for the file
            st.download_button('Download PDF', data=pdf_data, file_name='file.pdf')
            # Custom scores section
        
            # load in pretrained lda model for 3 topics
            lda_model = joblib.load('./data/3_topics_1200_reports_lda_model.pkl')

            #initialise empty dataframe to store probabilities in
            a = np.empty((df.shape[0],3))
            a[:] = np.nan
            df_probs = pd.DataFrame(columns=['S','G','E'], data=a)

            # iterate through clean paras and score the paragraph
            for index, row in df.iterrows():
                #get the topic probabilities
                test_bow = lda_model.id2word.doc2bow(df.clean_paras[index])
                res = lda_model[test_bow]
                #append probabilities to the df_probs
                for i in res:
                    row_num = index
                    col_num = i[0]
                    probability = i[1]
                    df_probs.iloc[row_num, col_num] = probability
            df_probs = pd.concat([df[['raw_paras','clean_paras']],df_probs],axis=1)
            df_probs = df_probs[df_probs['clean_paras'].apply(lambda x: len(x))>5]
        
            df_scores = df_probs[['S','G','E']]
            #drop any missing values to be safe
            df_scores = df_scores.dropna()
            #identify threshold for outliers
            p99_e = np.percentile(df_scores['E'], 99)

            # remove outliers
            df_scores_outliers_rem = df_scores[df_scores["E"] <= p99_e]

            #identify threshold for outliers
            p99_s = np.percentile(df_scores['S'], 99)
            # remove outliers
            df_scores_outliers_rem = df_scores_outliers_rem[df_scores_outliers_rem["S"] <= p99_s]

            #identify threshold for outliers
            p99_g = np.percentile(df_scores['G'], 99)
            # remove outliers
            df_scores_outliers_rem = df_scores_outliers_rem[df_scores_outliers_rem["G"] <= p99_g]

            # applying min_max scaler to scores
            scaler = MinMaxScaler()
            df_scores_fin= scaler.fit_transform(df_scores_outliers_rem)
            df_scores_fin = pd.DataFrame(df_scores_fin,columns=df_scores.columns)
            #finding the mean score for the report
            df_scores_fin_total = df_scores_fin[['G','E','S']].mean().T
            final_g_score = round(df_scores_fin_total['G']*100,2)
            final_e_score = round(df_scores_fin_total['E']*100,2)
            final_s_score = round(df_scores_fin_total['S']*100,2)

            st.write(f'Environmental Score = {final_e_score}')
            st.write(f'Social Score = {final_s_score}')
            st.write(f'Governance Score = {final_g_score}')



        #Printing the most relevant paragraph for each category
        greater_than_10 = df_probs[df_probs['clean_paras'].apply(lambda x: len(x))>10]
        top3e = greater_than_10.sort_values('E',ascending=False)[:3]
        st.subheader('Most Important Environmental Related Paragraph:')
        st.write(top3e.iloc[0,0])

        top3s = greater_than_10.sort_values('S',ascending=False)[:3]
        st.subheader('Most Important Social Related Paragraph:')
        st.write(top3s.iloc[0,0])

        top3g = greater_than_10.sort_values('G',ascending=False)[:3]
        st.subheader('Most Important Governance Related Paragraph:')
        st.write(top3g.iloc[0,0])


        #establish bag of words for uploaded pdf
        dictionary = corpora.Dictionary(df_probs.clean_paras)
        dictionary.filter_extremes(no_above=0.9)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in df.clean_paras]
        # trial between 5 and 15 topics to see which gives max coherence
        with st.spinner("Compiling topics, please wait..."):
            topic_num, max_coherence  = cp.get_max_coherence(doc_term_matrix,dictionary,df_probs.clean_paras,list(range(3,10)))
        #retrain model with optimal topic number
        Lda = gensim.models.ldamodel.LdaModel
        ldamodel = Lda(doc_term_matrix, num_topics=topic_num, id2word = dictionary, passes=40,\
                    iterations=200,  chunksize = 10000, eval_every = None, random_state=0)
        topic_data =  pyLDAvis.gensim_models.prepare(ldamodel, doc_term_matrix, dictionary, mds = 'pcoa')
        all_topics = {}
        num_terms = 10 # Adjust number of words to represent each topic
        lambd = 0 # Adjust this accordingly based on tuning above
        for i in range(1,topic_num+1): #Adjust this to reflect number of topics chosen for final LDA model
            topic = topic_data.topic_info[topic_data.topic_info.Category == 'Topic'+str(i)].copy()
            topic['relevance'] = topic['loglift']*(1-lambd)+topic['logprob']*lambd
            all_topics['Topic '+str(i)] = topic.sort_values(by='relevance', ascending=False).Term[:num_terms].values
        key_topics_df = pd.DataFrame(all_topics).T

        st.subheader("Key Topics and Relevant Words:")
        st.write(key_topics_df)

        html_string = pyLDAvis.prepared_data_to_html(topic_data)
        from streamlit import components
        components.v1.html(html_string, width=1300, height=800, scrolling=True)

