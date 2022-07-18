# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 19:50:56 2021

@author: Carlos Levi
"""
# Core packages
import streamlit as st

# EDA packages
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
def load_data(data): 
    df = pd.read_csv(data)
    return df

# Vectorize and cosine similarity
def vectorize_text(data):
    count_vec = CountVectorizer()
    cv_mat = count_vec.fit_transform(data)
    # now the cosine
    sim_mat = cosine_similarity(cv_mat)
    return sim_mat

# Content-based recommendation system
def get_recommendation(title,sim_mat,df,num_of_rec=5):
    # indices of the movie
    movie_indices = pd.Series(df.index,index=df['title']).drop_duplicates()
    # index of the movie
    idx = movie_indices[title]
    
    # Look into the cosine matrix for index
    sim_scores = list(enumerate(sim_mat[idx]))
    sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
    selected_movie_indices = [i[0] for i in sim_scores[1:]]
    selected_movie_scores = [i[0] for i in sim_scores[1:]]
    
    # Get the dataframe and title
    result_df = df.iloc[selected_movie_indices]
    result_df['similarity_score'] = selected_movie_scores
    final_recommend_movies = result_df[['title','similarity_score','genres','Year', 'rating']]
    return final_recommend_movies.head(num_of_rec)

# Search for a movie
#@st.cache 
def search_term_if_not_found(term,df):
    result_df = df[df['title'].str.contains(term)]
    return result_df

def main():
    st.title('Movie Recommendation App')
    menu = ['Popular Movies','Recommend','About']
    choice = st.sidebar.selectbox('Menu', menu)
    df = load_data('data.csv')
    if choice == 'Popular Movies':
        st.subheader('Popular Movies')
        dx = df[df['rating_counts'] >= 23000]
        dx.sort_values(by="rating_counts", ascending=False)
        st.dataframe(dx[['title','genres','Year', 'rating']])
    elif choice == 'Recommend':
        st.subheader('Recommend movies')
        sim_mat = vectorize_text(df['title'].values.astype('U'))
        search_term = st.text_input('Search')
        num_of_rec = st.sidebar.number_input('Number', 4,30,7)
        if st.button('Recommend'):
            if search_term is not None:
                try:
                    results = get_recommendation(search_term, sim_mat, df, num_of_rec)
                    st.dataframe(results)
                
                except:         
                    results = 'Not Found'
                    st.warning(results)
                    st.info('Suggested Options include')
                    result_df = search_term_if_not_found(search_term, df)
                    st.dataframe(result_df)
        
    else:
        st.subheader("About")
        st.text("Built with Streamlit & Pandas")
        
if __name__ == "__main__":
    main()
      

