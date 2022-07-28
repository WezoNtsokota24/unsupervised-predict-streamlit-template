"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Changing the background
import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
# Applying the function
set_png_as_page_bg('resources/imgs/Back_F.jpg')


# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Home","Exploratory Data Analysis","Recommender System","Solution Overview","About The App"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")

        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")

    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        title_SO = """
	    <div style="background-color:#464e5f00;padding:10px;border-radius:10px;margin:10px;border-style:solid; border-color:#000000; padding: 1em;">
	    <h1 style="color:black;text-align:center;">Solution Overview</h1>
        """
        st.markdown(title_SO, unsafe_allow_html=True)
        #st.title("Solution Overview")
        st.image('resources/imgs/Machine_learning.png',use_column_width=True)
        st.write("Describe your winning approach on this page")
        st.write("Our mission for this project is to construct a recommendation algorithm based on content or collaborative filtering, capable of accurately predicting how a user will rate a movie they have not yet viewed based on their historical preferences. Providing an accurate and robust solution to this challenge has immense economic potential, with users of the system being exposed to content they would like to view or purchase - generating revenue and platform affinity, so by following the Machine learning process above we have found a possible solution to the problem.")
        st.write("Source - The data for the MovieLens dataset is maintained by the GroupLens research group in the Department of Computer Science and Engineering at the University of Minnesota. Additional movie content data was legally scraped from IMDB")
        st.write("We used the MovieLens dataset. Below is a brief description of the dataset")
        st.write("Supplied Files: ")
        st.write("genome_scores.csv - a score mapping the strength between movies and tag-related properties.")
        st.write("genome_tags.csv - user assigned tags for genome-related scores.")
        st.write("imdb_data.csv - Additional movie metadata scraped from IMDB using the links.csv file.")
        st.write("links.csv - File providing a mapping between a MovieLens ID and associated IMDB and TMDB IDs.")
        st.write("sample_submission.csv - Sample of the submission format for the hackathon.")
        st.write("tags.csv - User assigned for the movies within the dataset.")
        st.write("test.csv - The test split of the dataset. Contains user and movie IDs with no rating data.")
        st.write("train.csv - The training split of the dataset. Contains user and movie IDs with associated rating data.")
        st.write("We followed the machine learning process which are Data collection, Data cleaning, Exploratory Data Analysis, Model building and Model deployment but the data collection process was already done for us by the GroupLens research group in the Department of Computer Science and Engineering at the University of Minnesota. so we  started with data cleaning where we looked for duplicates and we found that there was no duplicates in all the files and we looked for missing values then we saw that only the df_links and df_tags have null values from the tmdId the df_links has 107 null values while the tag column from the df_tags is missing about 16 values these values are so less that we can even drop them and won't lose any information. We also saw that The df_train dataset has no missing values and the df_imdb dataset is missing about 71.02 percent of the budget column which we can't replace by the mean reliable since we missing alot of data and 36.20 percent from the director column is missing which we can't impute directors reliably same with the plot_keywords which is missing about 40.61 percent of the data and the the runtime is missing about 44.32 percent of the data. and the title_cast column is missing about 36.91 percent of the data most columns can't be imputed reliably. and we can't use the budget column as one of our features since it is missing a lot of values, thus it is advisable to drop the budget column. ")
        st.write("After data preprocessing we started by building our models. We built six different collaborative base models, namely SVD, SVDpp, Normal Predictor, CoClustering, KNN Baseline, and Non-Negative Matrix Factorization (NMF). And we build a content based model using the director, title_cast, plot_keywords and genres are the features. ")

        imdb = """
	    <div style="background-color:#464e5f00;padding:10px;border-radius:10px;margin:10px;">
	    <h3 style="color:black;text-align:left;">Cleaning the imdb_data dataset</h3>
        """
        st.markdown(imdb, unsafe_allow_html=True)
        st.write('We imputed the runtime with the mean runtime.\n\nCreated a list plot keywords for each movie.\n\nCreated a list of title casts for each movie. \n\n We imputed the title_cast column with the value= "Unknown cast". \n\n We imputed the plot_keywords column with the value= "Unknown keywords".\n\n We imputed the director column with the value= "Unknown director".')

        movies = """
	    <div style="background-color:#464e5f00;padding:10px;border-radius:10px;margin:10px;">
	    <h3 style="color:black;text-align:left;">Cleaning the movies dataset</h3>
        """
        st.markdown(movies, unsafe_allow_html=True)
        st.write('Created a list of genres in every movie in the movies column\n\nThe release_year column was added.')

        train = """
	    <div style="background-color:#464e5f00;padding:10px;border-radius:10px;margin:10px;">
	    <h3 style="color:black;text-align:left;">Cleaning the train dataset</h3>
        """
        st.markdown(train, unsafe_allow_html=True)
        st.write('We dropped the timestamp column since it was no use to us.')




        st.write('After cleaning the data, we then merged the ')
        st.write('We proceeded to the third step, the Exploratory Data Analysis. We constructed various visualisations using our data and gathered insights from our data, these are on our Exploratory Data Analysis page.')

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.

    # Home page
    if page_selection == "Home":
        st.image("resources/imgs/logo.png")
    
        #About
    if page_selection == "About The App":
        title_about = """
	    <div style="background-color:#464e5f00;padding:10px;border-radius:10px;margin:10px;">
	    <h1 style="color:black;text-align:center;">  The Team </h1>
        <h3 style="color:black;text-align:right;">We are Team CBB3 from Explore Data Science Academy(BIOSKOP HANDLERS). This is our  movie recommender project for 2022 July unsupervised sprint.</h3>
        """
        mission = """
	    <div style="background-color:#464e5f00;padding:10px;border-radius:10px;margin:10px;">
	    <h1 style="color:black;text-align:center;">  Our Objective  </h1>
        <h3 style="color:black;text-align:center;">To ensure great movie experience, by giving you the client 90-99% match to what you like in movie selection. </h3>
        """

        contributors = """
        <div style="background-color:#464e5f00;padding:10px;border-radius:10px;margin:10px;">
	    <h1 style="color:black;text-align:center;">  Meet The Team </h1>
        <h3 style="color:black;text-align:center;">Sinoluthando Bokaba</h3>
        <h3 style="color:black;text-align:center;">Wezo Ntsokota</h3>
        <h3 style="color:black;text-align:center;">Nthapeleng Linah Raphela</h3>
        <h3 style="color:black;text-align:center;">Vhumboni Mildred Maluleke</h3>
        <h3 style="color:black;text-align:center;">Sedibo Lefa Matlala</h3>
        <h3 style="color:black;text-align:center;">Chris Barnett (Supervisor)</h3>
        """
        st.markdown(title_about, unsafe_allow_html=True)
        st.markdown(mission, unsafe_allow_html=True)
        st.markdown(contributors, unsafe_allow_html=True)

    # EDA
    if page_selection == "Exploratory Data Analysis":
        eda_title = """
	    <div style="background-color:#464e5f00;padding:10px;border-radius:10px;margin:10px;border-style:solid; border-color:#000000; padding: 1em;">
	    <h1 style="color:black;text-align:center;">Exploratory Data Analysis</h1>
        """
        st.markdown(eda_title, unsafe_allow_html=True)
        #st.image('resources/imgs/EDA.png',use_column_width=True)

        # available options = ["Ratings", "Genre", "Director", "Movies", "Actors"]

        sys_eda = st.radio("Choose an EDA section",
        ('Ratings','Movies','Directors','Genres','Actors'))
        # Ratings option
        if sys_eda == "Ratings":


            op_ratings = st.radio("Choose an option under ratings",("User Rating distribution","Average rating per genre"))

            if op_ratings == "User Rating distribution":
                st.image("resources/imgs/Ratings_Distribution_from_Users.png")
                st.write("The ratings seems to be following a normal distribution with a mean of only 3.533395 and a median of 3.500000 and a mode of 4.000 but maximum values reaching as high as 5, which is almost 2 times greater than the mean. However, as can be seen from the distribution plot, almost all movies have a ratings score less than 5 (the 75th percentile is at 4.000000). It appears that Users are extremely generous in their ratings. The mean rating is only a 3.533395 on a scale of 5. Half the movies have a rating of greater than or equal to 3.0.")

            if op_ratings == "Average rating per genre":
                st.image("resources/imgs/the_average_rating_per_genre_using_a_box-plot.png")
                st.write("Almost all the genres had the same distribution for rating on the box plot except for the  Firm Noir one which has a bigger distribution than the rest for some reason I haven't figured right now. ")

        # Genres option
        if sys_eda == "Genres":
        
            op_genre = st.radio("Choose an option under Genres",("Treemap of movie genres","Runtime per genre"))
            

            if op_genre == "Treemap of movie genres":
            
                st.image('resources/imgs/Treemap_G1.png',use_column_width=True)
                st.write("The dominance of drama as a genre is not surprising when we consider the following:Drama is the cheapest genre to produce as movies don’t necessarily require special sets, costumes, locations, props, special/visual effects, etc. Drama has the broadest definition of all genres – everything that happens anywhere ever is a drama. Conversely, other genres have a higher bar for classification, such as the need for high-octane events for a movie to be classed as Action, scary events to be Horror, funny elements to be a Comedy, etc")


            if op_genre == "Runtime per genre":
                st.image("resources/imgs/Average_runtime.png")
                st.write("we can see that movies for children had a shorter runtime than the rest and which makes sense since children have a short concentration span than adults ")


        # Directors option
        if sys_eda == "Directors":

            st.info("Even though they may not appear in front of the camera, the director is one of the most important people on a film set. They do more than shout “action” and “cut” behind the scenes—they’re the person who determines the creative vision and makes all of the film’s biggest decisions.")
            #op_director = st.radio("Choose an option under directors"),("Top 10 popular directors ")

            st.image("resources/imgs/Top_10_Most_Popular_Movie_Directors.png")
            st.write("we can see william shakespeare as one of the popular directors which already shows that this dataset has alot of old movies. ")

        # Movies option
        if sys_eda == "Movies":
            op_movie = st.radio("Choose an option under movies",("Total movies released per year","All time Popular Movies by ratings","Wordcloud of the titles of the movies"))


            if op_movie == "Total movies released per year":
                st.image("resources/imgs/Total_movies_released_per_year.png")
                st.write("It appears that 2015, 2016, 2014, 2017 and 2013 are the most popular years when it comes to movie releases in that order. In Hollywood circles, this is also known as the the dump years when sub par movies are released by the dozen. Global film production is booming, thanks in large part to new technologies. It’s cheaper and easier than ever before to shoot, edit and distribute a feature film, not to mention the effect of the internet in sharing ideas, knowledge and advice. It’s not possible to give a definitive figure for the exact number of films made each year but the growth can be seen from the graph above. And the consumption of movies is way higher than before so to meet the demand then more movies are released.")

            if op_movie == "All time Popular Movies by ratings":
                st.image("resources/imgs/Top_15_All_time_Popular_Movies_by_ratings.png")
                st.write("The Shawshank Redemption(1994) Is a hollywood classic. It is a simple movie with a deep and everlasting message. Not only do the performance of Freeman and Robbins rank among the best of all time, but Shawshank is filled with brilliantly realised supporting characters who surprise and enthrall in equal measures. Tim Robbins and Morgan Freeman have given an outstanding performance which enhance the overall impact of the movie. Meanwhile, Pulp Fiction and The Shawshank Redemption each received seven awards. Forrest Gump is a timeless classic and it is deservedly so. It is the perfect movie to watch when you're in the mood for a little soul-searching. The story about one man's incredible and unexpected life journey is as significant now as it was when the film was first released in 1994. This movie is superbly acted, has great themes, some hilarious humor, a well written and interesting story, beautiful music by Alan Silvestri, a fantastic late twentieth century themed soundtrack and meaningful characters. The movie won the best picture Oscar, earned 677 million dollars around the world and is hailed by many as a modern classic filled with homespun catchphrases like 'My momma always said life was like a box of chocolates. you never know what you're gonna get'")

            if op_movie == "Wordcloud of the titles of the movies":
                st.image("resources/imgs/title_wordcloud.png")
                st.write("The word Love is the most commonly used word in movie titles. Girl, Boy, Woman and Man are also among the most commonly occuring words. I think this encapsulates the idea of the presence of romance in movies pretty well.")

        if sys_eda == "Actors":

            op_actors = st.radio("Choose an option under actors",("Title cast wordcloud","Top 20 popular actors"))

            if op_actors == "Title cast wordcloud":
                st.image("resources/imgs/titlecast_wordcloud.png")
                st.write("About 36.91 percent of the title cast were missing we can see that a lot of our movies and we imputed those unknown cast for movies with the string 'Unknown cast' now we only considered a insights on the wordcloud  without the unknown cast strings below for better observation on what we already have.we can see the names of the cast such as Michael, David, Paul, James, John, Peter, Richard and Robert just to name a few these names had a large fontsize because they appeared a lot on the column meaning these actors must have been in most movies from our dataset.")

            if op_actors == "Top 20 popular actors":
                st.image("resources/imgs/Top_20_Popular_Actors.png")
                st.write("Choosing Cast is a little more tricky. Lesser known actors and minor roles do not really affect people's opinion of a movie. Therefore, we must only select the major characters and their respective actors. Arbitrarily we will choose the top popular actors that appear in the movies: Samuel L.jackson The best Samuel L. Jackson movies all have one thing in common… They all feature a certain bad person who can do good at times and they can also do bad. He’s been in countless movies over the years, but we wanted to find out what were his best roles. To that end, fans like you have voted on their favorite Samuel L. Jackson movies are Pulp Fiction and Django Unchained(2012) ")




if __name__ == '__main__':
    main()
