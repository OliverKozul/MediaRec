import os
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from dotenv import load_dotenv
from werkzeug.security import check_password_hash
from core.utils import summarize_user_preferences
from models.media import Media
from models.user import User
from recommender.engine import RecommendationEngine
from core.logger import Logger

load_dotenv()
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.getenv("SECRET_KEY")
logger = Logger().get_logger("WebApp")

recommendation_engine = RecommendationEngine()
recommendation_engine.load_model()
recommendation_engine.load_vectorizers()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.get_user(username=username)
        if user and user['password'] and check_password_hash(user['password'], password):
            session['user_id'] = user['user_id']
            session['username'] = user['username']
            flash('Logged in successfully!', 'success')
            logger.info(f"User '{username}' logged in successfully.")
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))

        user_id = User.create_user(username, password)
        if user_id['success']:
            flash('Registration successful! Please log in.', 'success')
            logger.info(f"User '{username}' registered successfully.")
            return redirect(url_for('login'))
        else:
            flash('Username already exists', 'danger')

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    logger.info("User logged out.")
    return redirect(url_for('index'))

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    temperature = float(request.args.get('temperature', 0))
    recommendations = recommendation_engine.recommend(
        user_id=session['user_id'], 
        top_n=10,
        temperature=temperature
    )

    return render_template('index.html', 
                        recommendations=recommendations,
                        current_temperature=temperature)

@app.route('/media/<int:media_id>')
def media_details(media_id: int):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    temperature = float(request.args.get('temperature', 0))

    media = Media.get_media(media_id)
    user_rating = User.get_ratings(session['user_id']).get(media_id)
    user_rating_count = len(User.get_ratings(session['user_id']))
    media_dict = dict(zip([
        "media_id", "title", "media_type", "genres", "description", "director", "actors", "poster_path"
    ], media)) if media else {}
    media_df = pd.DataFrame([media_dict]) if media_dict else pd.DataFrame()

    if user_rating_count >= 1 and not media_df.empty:
        explanation = recommendation_engine.explain_single_prediction(session['user_id'], media_df, temperature=temperature)
    else:
        explanation = None

    return render_template('media.html', media=media_dict, user_rating=user_rating, explanation=explanation, temperature=temperature)

@app.route('/rate_media', methods=['POST'])
def rate_media():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    data = request.get_json()
    media_id = data['media_id']
    media_title = data['media_title']
    rating = int(data['rating'])

    try:
        User.rate(session['user_id'], media_id, rating)
        logger.info(f"User {session['user_id']} rated media {media_id} ({media_title}) with rating {rating}.")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error rating media {media_id} ({media_title}) for user {session['user_id']}: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    rated_media = User.get_all_media_for_user(session['user_id'])
    rated_media = [
        {
            **media_info
        }
        for media_info in rated_media
    ]
    preferences = summarize_user_preferences(user_id=session['user_id'])
    return render_template('profile.html', 
                        rated_media=rated_media,
                        preferences=preferences)

@app.route('/delete-all-ratings', methods=['POST'])
def delete_all_ratings():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    try:
        User.delete_all_ratings(session['user_id'])
        flash('All your ratings have been deleted', 'success')
    except Exception as e:
        flash('Error deleting ratings', 'danger')
        logger.error(f"Error deleting ratings: {e}")
    return redirect(url_for('profile'))

@app.route('/delete_rating/<int:media_id>', methods=['POST'])
def delete_rating(media_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    try:
        User.delete_rating(session['user_id'], media_id)
        flash('Rating removed successfully', 'success')
    except Exception as e:
        flash('Error removing rating', 'danger')
    return redirect(url_for('profile'))

@app.route('/search')
def search_media():
    query = request.args.get('query', '').lower()
    if not query:
        return redirect(url_for('index'))
    
    all_media = Media.get_all_media()
    media = [{
        "media_id": m[0],
        "title": m[1],
        "media_type": m[2],
        "genres": m[3],
        "description": m[4],
        "director": m[5],
        "actors": m[6],
        "poster_path": m[7]
    } for m in all_media]
    results = []

    for item in media:
        match_reasons = []

        if item['director'] and query in item['director'].lower():
            match_reasons.append('Director')

        actors = [a.strip().lower() for a in item['actors'].split(',')] if item['actors'] else []
        if any(query in a for a in actors):
            match_reasons.append('Actor')

        if query in item['title'].lower():
            match_reasons.append('Title')

        if item.get('description') and query in item['description'].lower():
            match_reasons.append('Description')

        if match_reasons:
            item['match_reasons'] = match_reasons
            results.append(item)

    return render_template('search_results.html', 
                         query=query,
                         media=results)

if __name__ == '__main__':
    # initialize_db() # For testing purposes only
    app.run(host="0.0.0.0", debug=True)