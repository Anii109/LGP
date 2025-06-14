<<<<<<< HEAD
# LGP-Project
Machine Learning 
=======
# LGP
Machine Learning -Movie recommender system
>>>>>>> 
A simple content-based movie recommendation system that suggests movies similar to the one you search for. Built using Python and Streamlit.

>>>Files Involoved <<<<<
ml.py` → Python file to process data and generate required `.pkl` files  
app.py` → Streamlit web app file  
tmdb_5000_movies.csv` → Dataset containing movie details  
tmdb_5000_credits.csv` → Dataset containing cast and crew information  
movies.pkl` → Generated file containing movie features  
similarity.pkl` → Generated file containing cosine similarity matrix  


>>>>>How to Run<<<<
To run this Movie Recommender System, first clone the repository using git clone . After cloning, ensure you have Python installed along with all the required libraries (pandas, numpy, sklearn, nltk, ast, pickle, and streamlit). Start by running the ml.py file which will process the movie dataset and automatically generate two important files: movies.pkl and similarity.pkl. These files are essential for making recommendations. Once generated, you can then run the app.py file using the command streamlit run app.py. This will launch the web app in your browser where you can search for a movie and get similar recommendations.