# MediaRec

A movie recommender system created as part of an Artificial Intelligence university project.

## Features

- Movie recommendation engine  
- User profiles  
- Search functionality  
- Data processing and integration with TMDB  
- Web interface (HTML/CSS/JS)  

## Folder Structure

- `core` : Database setup, logging, utilities  
- `data` : Data processing and loading  
- `models` : Data models (media, user)  
- `recommender` : Recommendation engine and training  
- `static` : Frontend assets (CSS, JS)  
- `templates` : HTML templates  

## Installation

1. Clone the repository:
   
   ```bash
   git clone https://github.com/OliverKozul/MediaRec.git
   ```
2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```
3. Run the app:

   ```bash
   python app.py
   ```

## Requirements

To run this project locally, you need the following:

- Python 3.10+ (Tested with Python 3.10.6)
- See requirements.txt for dependencies

# Implementation Details

The following sections the steps I took when implementing this project.

## Data

To train the model used to predict user preferences, the [MovieLens 32M Dataset](https://grouplens.org/datasets/movielens/) was used.
The number of samples is excessive for the scope of this project and the capabilities of my computer but the data was up-to-date and proved extremely useful.

MovieLens provided three tables: 
  - movies.csv listed movies, with their respective IDs, titles and genres.
  - links.csv linked the movie IDs with their IMDB and TMDB IDs.
  - ratings.csv contained 32 million user ratings which were used for model training.

### Enrichment

For the model to gain a broader context of user preferences, I wanted to provide it with other features except for movie genres.
To this end, I utilized TMDB's generous developer API to fetch movie descriptions, actors and directors.
Additionally, movie posters were also saved to improve the user experience with visuals in the final web application.

### Preprocessing

Having gathered all the necessary data, I proceeded by cleaning and preprocessing the dataset.
Each movie entry in the dataset listed many genres and actors, yet limiting both categories to the top three values proved to be sufficient and reduced noise.
Even after removing stop words, movie descriptions appeared to cause worse results during training and testing, so they were removed as a feature.

## Training

After data preprocessing, the last step was to merge the movie and rating dataset by the movie IDs, therefore obtaining the final dataset used in model training.
Before settling for a XGBoost regression model, I initially experimented with random forest regressors and a deep learning approach.

The neural network was too computationally intense and did not provide substantial benefit, thus it was not a viable solution.
Random forest offered slightly worse results and longer training times when compared to XGBoost, making XGBoost the obvious choice in this project.

The goal of the model was to minimize the error when predicting a user rating given the previously mentioned features.
Modifying hyperparameters did not alter the results significantly, except for n_jobs, which decreased training time drastically by parallelizing the process.
The main bottleneck during this phase was RAM, as the datasets utilized were extremely large.

## Web UI

To provide a pleasant user experience, a simple web application was created.
This part of the project was not my main focus, therefore I did not implement many features as it primarily served as a medium to display results.
I allowed the user to register and login on the website and by rating movies using the profile the model would gain insight into the user's preferences.
Additionally, the user can adjust a temperature slider, a scaling factor to bias ranking diversity, on the home page that affects the ranking score of the content.
Decreasing the temperature gives higher precedence to movies similar to the user's preferences.
In contrast, increasing the temperature leads to a more diverse selection of movies that should still be appealing to the user.

## Project Structure

- As this project was written in Python, I used Flask for my web framework.
- The user and movie data is stored in a local SQLite database.
- I defined a user and media class that serve as an interface with the database.

All the previously mentioned systems work together to offer the user a useful tool when looking for new viewing content.
