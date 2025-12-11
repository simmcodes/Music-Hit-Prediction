"""
Music Hit Classifier - Streamlit Web App
Interactive interface for predicting song success
Uses ACTUAL model metrics from training
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Page configuration
st.set_page_config(
    page_title="Music Hit Classifier",
    page_icon="🎵",
    layout="wide"
)

# Load the trained model AND metrics
@st.cache_resource
def load_model_and_metrics():
    """Load the pre-trained Random Forest model and performance metrics"""
    try:
        model = joblib.load('models/best_model.pkl')
        metrics = joblib.load('models/model_metrics.pkl')
        return model, metrics
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {str(e)}")
        st.info("Please run the Jupyter notebook to train the model and save metrics first.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading files: {str(e)}")
        return None, None

# Title and description
st.title("🎵 Music Hit Classifier")

# Load model and metrics
model, metrics = load_model_and_metrics()

# Show dynamic accuracy if available
if metrics:
    accuracy = metrics['accuracy']
    st.markdown(f"""
    Predict whether a song will become a **hit** based on its audio features and artist popularity.
    This tool uses a Random Forest model trained on 150 songs with **{accuracy:.1%} accuracy**.
    """)
else:
    st.markdown("""
    Predict whether a song will become a **hit** based on its audio features and artist popularity.
    This tool uses a Random Forest model trained on Spotify data.
    """)
    st.warning("⚠️ Model metrics not loaded. Train the model in Jupyter notebook first.")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🎯 Single Prediction", "📈 Model Info"])

#═══════════════════════════════════════════════════════════
# PAGE 1: HOME
#═══════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.header("Welcome to the Music Hit Classifier!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if metrics:
            st.metric("Model Accuracy", f"{metrics['accuracy']:.1%}", "Tested on Real Data")
        else:
            st.metric("Model Accuracy", "N/A", "Train model first")
    
    with col2:
        if metrics:
            st.metric("Test Songs", f"{metrics['test_size']}", "Evaluation Set")
        else:
            st.metric("Training Songs", "150", "Diverse Genres")
    
    with col3:
        if metrics:
            st.metric("Features Used", f"{metrics['num_features']}", "Audio + Artist")
        else:
            st.metric("Features Used", "16", "Audio + Artist")
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📊 What This Tool Does")
    st.write("""
    - **Predicts** if a song will become a hit (popularity ≥ 70)
    - **Analyzes** 16 features including audio characteristics and artist metrics
    - **Identifies** "hidden gems" - quality songs with low visibility
    - **Helps** A&R teams discover talented artists early
    """)
    
    st.subheader("🎯 How to Use")
    st.write("""
    1. **Single Prediction**: Enter song features manually to get instant prediction
    2. **Model Info**: Learn about the model's performance and feature importance
    """)
    
    # Sample prediction
    st.markdown("---")
    st.subheader("🎬 Try Quick Examples")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎵 Example: Major Artist Hit"):
            if model:
                example_data = pd.DataFrame({
                    'danceability': [0.85],
                    'energy': [0.92],
                    'loudness': [-4],
                    'speechiness': [0.05],
                    'acousticness': [0.08],
                    'instrumentalness': [0.0],
                    'liveness': [0.15],
                    'valence': [0.82],
                    'tempo': [125],
                    'artist_followers': [15000000],
                    'artist_popularity': [88],
                    'lastfm_playcount': [20000000],
                    'lastfm_listeners': [1500000],
                    'quality_index': [0.863],
                    'visibility_score': [0.00133],
                    'artist_reach_millions': [15.0]
                })
                
                prediction = model.predict(example_data)[0]
                confidence = model.predict_proba(example_data)[0]
                
                if prediction == 1:
                    st.success(f"🎉 **PREDICTION: HIT!** (Confidence: {confidence[1]*100:.1f}%)")
                else:
                    st.warning(f"😕 **PREDICTION: NOT HIT** (Confidence: {confidence[0]*100:.1f}%)")
    
    with col2:
        if st.button("🎸 Example: Slipknot - Wait and Bleed"):
            if model:
                example_data = pd.DataFrame({
                    'danceability': [0.382],
                    'energy': [0.996],
                    'loudness': [-4.119],
                    'speechiness': [0.104],
                    'acousticness': [0.00208],
                    'instrumentalness': [0.0],
                    'liveness': [0.417],
                    'valence': [0.327],
                    'tempo': [93.345],
                    'artist_followers': [14294483],
                    'artist_popularity': [80],
                    'lastfm_playcount': [17775195],
                    'lastfm_listeners': [1574487],
                    'quality_index': [0.5333],
                    'visibility_score': [1.2435],
                    'artist_reach_millions': [14.294483]
                })
                
                prediction = model.predict(example_data)[0]
                confidence = model.predict_proba(example_data)[0]
                
                if prediction == 1:
                    st.success(f"🎉 **PREDICTION: HIT!** (Confidence: {confidence[1]*100:.1f}%)")
                    st.info("This is a real hit song from your dataset!")
                else:
                    st.warning(f"😕 **PREDICTION: NOT HIT** (Confidence: {confidence[0]*100:.1f}%)")

#═══════════════════════════════════════════════════════════
# PAGE 2: SINGLE PREDICTION
#═══════════════════════════════════════════════════════════
elif page == "🎯 Single Prediction":
    st.header("🎯 Single Song Prediction")
    st.write("Enter the features of a song to predict if it will be a hit.")
    
    # Add preset buttons at the top
    st.markdown("### 🎯 Quick Load Presets")
    col1, col2, col3 = st.columns(3)
    
    preset_loaded = False
    
    with col1:
        if st.button("🎸 Famous Rock Band (HIT)", use_container_width=True):
            st.session_state['preset'] = 'rock'
            preset_loaded = True
    
    with col2:
        if st.button("🎤 Indie Artist (NOT HIT)", use_container_width=True):
            st.session_state['preset'] = 'indie'
            preset_loaded = True
    
    with col3:
        if st.button("🎵 Pop Star (HIT)", use_container_width=True):
            st.session_state['preset'] = 'pop'
            preset_loaded = True
    
    # Set defaults based on preset
    if 'preset' not in st.session_state:
        st.session_state['preset'] = 'default'
    
    if st.session_state['preset'] == 'rock':
        defaults = {
            'danceability': 0.38, 'energy': 0.996, 'valence': 0.33, 'tempo': 93,
            'loudness': -4.1, 'acousticness': 0.002, 'speechiness': 0.10,
            'instrumentalness': 0.0, 'liveness': 0.42,
            'artist_followers': 14294483, 'artist_popularity': 80,
            'lastfm_playcount': 17775195, 'lastfm_listeners': 1574487
        }
    elif st.session_state['preset'] == 'indie':
        defaults = {
            'danceability': 0.5, 'energy': 0.6, 'valence': 0.4, 'tempo': 100,
            'loudness': -8.0, 'acousticness': 0.5, 'speechiness': 0.05,
            'instrumentalness': 0.1, 'liveness': 0.2,
            'artist_followers': 50000, 'artist_popularity': 35,
            'lastfm_playcount': 5000, 'lastfm_listeners': 2000
        }
    elif st.session_state['preset'] == 'pop':
        defaults = {
            'danceability': 0.85, 'energy': 0.92, 'valence': 0.82, 'tempo': 125,
            'loudness': -4.0, 'acousticness': 0.08, 'speechiness': 0.05,
            'instrumentalness': 0.0, 'liveness': 0.15,
            'artist_followers': 15000000, 'artist_popularity': 88,
            'lastfm_playcount': 20000000, 'lastfm_listeners': 1500000
        }
    else:  # default - set to values that will predict HIT
        defaults = {
            'danceability': 0.80, 'energy': 0.85, 'valence': 0.75, 'tempo': 120,
            'loudness': -5.0, 'acousticness': 0.15, 'speechiness': 0.05,
            'instrumentalness': 0.0, 'liveness': 0.2,
            'artist_followers': 10000000, 'artist_popularity': 82,
            'lastfm_playcount': 15000000, 'lastfm_listeners': 1000000
        }
    
    st.markdown("---")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎼 Audio Features")
        
        danceability = st.slider("Danceability", 0.0, 1.0, defaults['danceability'], 0.01,
                                 help="How suitable for dancing (0=not danceable, 1=very danceable)")
        
        energy = st.slider("Energy", 0.0, 1.0, defaults['energy'], 0.01,
                          help="Intensity and activity (0=calm, 1=energetic)")
        
        valence = st.slider("Valence (Positivity)", 0.0, 1.0, defaults['valence'], 0.01,
                           help="Musical positivity (0=sad, 1=happy)")
        
        tempo = st.number_input("Tempo (BPM)", 50, 200, defaults['tempo'],
                               help="Speed in beats per minute")
        
        loudness = st.slider("Loudness (dB)", -60.0, 0.0, defaults['loudness'], 0.5,
                            help="Overall volume (-60=quiet, 0=loud)")
        
        acousticness = st.slider("Acousticness", 0.0, 1.0, defaults['acousticness'], 0.01,
                                help="How acoustic vs electronic")
        
        speechiness = st.slider("Speechiness", 0.0, 1.0, defaults['speechiness'], 0.01,
                               help="Amount of spoken words")
        
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, defaults['instrumentalness'], 0.01,
                                    help="Predicts if track has no vocals")
        
        liveness = st.slider("Liveness", 0.0, 1.0, defaults['liveness'], 0.01,
                            help="Presence of audience (0=studio, 1=live)")
    
    with col2:
        st.subheader("👤 Artist Metrics")
        
        artist_followers = st.number_input("Artist Followers", 0, 100000000, defaults['artist_followers'],
                                          help="Number of Spotify followers")
        
        artist_popularity = st.slider("Artist Popularity", 0, 100, defaults['artist_popularity'],
                                     help="Spotify popularity score (0-100)")
        
        lastfm_playcount = st.number_input("Last.fm Play Count", 0, 100000000, defaults['lastfm_playcount'],
                                          help="Total plays on Last.fm")
        
        lastfm_listeners = st.number_input("Last.fm Listeners", 0, 10000000, defaults['lastfm_listeners'],
                                          help="Unique listeners on Last.fm")
        
        st.markdown("---")
        st.info("""
        💡 **Tip:** To predict a HIT, you typically need:
        - Artist followers: 10M+
        - Artist popularity: 80+
        - High energy & danceability
        - Last.fm plays: 10M+
        """)
    
    # Calculate engineered features
    st.subheader("🔧 Auto-calculated Features")
    
    # Scale the audio features
    quality_index = (danceability + energy + valence) / 3
    visibility_score = lastfm_playcount / (artist_followers + 1) if artist_followers > 0 else 0
    artist_reach_millions = artist_followers / 1000000
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Quality Index", f"{quality_index:.2f}")
    col2.metric("Visibility Score", f"{visibility_score:.4f}")
    col3.metric("Artist Reach (M)", f"{artist_reach_millions:.2f}")
    
    # Predict button
    st.markdown("---")
    
    if st.button("🎵 PREDICT", type="primary", use_container_width=True):
        if model:
            # Prepare input data
            input_data = pd.DataFrame({
                'danceability': [danceability],
                'energy': [energy],
                'loudness': [loudness],
                'speechiness': [speechiness],
                'acousticness': [acousticness],
                'instrumentalness': [instrumentalness],
                'liveness': [liveness],
                'valence': [valence],
                'tempo': [tempo],
                'artist_followers': [artist_followers],
                'artist_popularity': [artist_popularity],
                'lastfm_playcount': [lastfm_playcount],
                'lastfm_listeners': [lastfm_listeners],
                'quality_index': [quality_index],
                'visibility_score': [visibility_score],
                'artist_reach_millions': [artist_reach_millions]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            confidence = model.predict_proba(input_data)[0]
            
            # Display result
            st.markdown("### 🎯 Prediction Result")
            
            if prediction == 1:
                st.success(f"""
                ## 🎉 HIT!
                
                **Confidence:** {confidence[1]*100:.1f}%
                
                This song has strong hit potential based on its features!
                """)
                
                # Show why
                st.info("""
                **Contributing Factors:**
                - Quality Index: {:.2f} (Feel-good factor)
                - Artist Reach: {:.2f}M followers
                - High Energy: {:.2f}
                - Artist Popularity: {}
                """.format(quality_index, artist_reach_millions, energy, artist_popularity))
                
            else:
                st.warning(f"""
                ## 😕 NOT A HIT
                
                **Confidence:** {confidence[0]*100:.1f}%
                
                Based on current features, this may not reach mainstream hit status.
                """)
                
                # Show why
                st.info("""
                **Possible Reasons:**
                - Artist has limited reach ({:.2f}M followers)
                - Quality Index: {:.2f} (could be higher)
                - Artist Popularity: {} (needs 80+)
                - May need more promotion or timing
                """.format(artist_reach_millions, quality_index, artist_popularity))
            
            # Feature contribution
            st.markdown("### 📊 Your Song's Profile")
            
            fig = go.Figure()
            
            features_display = {
                'Danceability': danceability,
                'Energy': energy,
                'Valence': valence,
                'Quality Index': quality_index,
                'Artist Popularity': artist_popularity/100
            }
            
            fig.add_trace(go.Bar(
                x=list(features_display.keys()),
                y=list(features_display.values()),
                marker_color=['#1DB954', '#1DB954', '#1DB954', '#FF6B6B', '#4A90E2']
            ))
            
            fig.update_layout(
                title="Key Feature Values (0-1 scale)",
                yaxis_title="Value",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ Model not loaded. Please train the model first.")

#═══════════════════════════════════════════════════════════
# PAGE 3: BATCH PREDICTION
#═══════════════════════════════════════════════════════════
elif page == "📊 Batch Prediction":
    st.header("📊 Batch Prediction")
    st.write("Upload a CSV file with song features to predict multiple songs at once.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV with columns: danceability, energy, valence, tempo, loudness, etc."
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} songs.")
            
            # Show preview
            with st.expander("👀 Preview Data (First 5 rows)"):
                st.dataframe(df.head())
            
            # Check required columns
            required_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                           'acousticness', 'instrumentalness', 'liveness', 'valence',
                           'tempo', 'artist_followers', 'artist_popularity',
                           'lastfm_playcount', 'lastfm_listeners']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Required columns: " + ', '.join(required_cols))
            else:
                # Add engineered features if not present
                if 'quality_index' not in df.columns:
                    df['quality_index'] = (df['danceability'] + df['energy'] + df['valence']) / 3
                
                if 'visibility_score' not in df.columns:
                    df['visibility_score'] = df['lastfm_playcount'] / (df['artist_followers'] + 1)
                
                if 'artist_reach_millions' not in df.columns:
                    df['artist_reach_millions'] = df['artist_followers'] / 1000000
                
                # Predict button
                if st.button("🎯 Predict All Songs", type="primary"):
                    if model:
                        with st.spinner("Making predictions..."):
                            # Select features for prediction
                            feature_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                                          'acousticness', 'instrumentalness', 'liveness', 'valence',
                                          'tempo', 'artist_followers', 'artist_popularity',
                                          'lastfm_playcount', 'lastfm_listeners',
                                          'quality_index', 'visibility_score', 'artist_reach_millions']
                            
                            X = df[feature_cols].fillna(0)
                            
                            # Make predictions
                            predictions = model.predict(X)
                            probabilities = model.predict_proba(X)
                            
                            # Add to dataframe
                            df['Prediction'] = ['HIT' if p == 1 else 'NOT HIT' for p in predictions]
                            df['Hit_Probability'] = probabilities[:, 1]
                            
                        st.success("✅ Predictions complete!")
                        
                        # Summary stats
                        col1, col2, col3 = st.columns(3)
                        
                        hits = sum(predictions == 1)
                        not_hits = sum(predictions == 0)
                        avg_confidence = probabilities[:, 1].mean()
                        
                        col1.metric("Predicted Hits", hits, f"{hits/len(df)*100:.1f}%")
                        col2.metric("Predicted Not Hits", not_hits, f"{not_hits/len(df)*100:.1f}%")
                        col3.metric("Avg Hit Probability", f"{avg_confidence:.2f}")
                        
                        # Show results
                        st.markdown("### 📋 Prediction Results")
                        
                        # Allow filtering
                        filter_option = st.radio("Filter", ["All", "Hits Only", "Not Hits Only"])
                        
                        if filter_option == "Hits Only":
                            display_df = df[df['Prediction'] == 'HIT']
                        elif filter_option == "Not Hits Only":
                            display_df = df[df['Prediction'] == 'NOT HIT']
                        else:
                            display_df = df
                        
                        # Show relevant columns
                        display_cols = ['track_name', 'artists', 'Prediction', 'Hit_Probability', 
                                       'quality_index', 'artist_followers'] if 'track_name' in df.columns else \
                                      ['Prediction', 'Hit_Probability', 'quality_index', 'artist_followers']
                        
                        display_cols = [col for col in display_cols if col in display_df.columns]
                        
                        st.dataframe(display_df[display_cols], use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Visualizations
                        st.markdown("### 📊 Visualizations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pie chart
                            fig = px.pie(
                                values=[hits, not_hits],
                                names=['Hit', 'Not Hit'],
                                title='Prediction Distribution',
                                color_discrete_sequence=['#1DB954', '#FF6B6B']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Probability distribution
                            fig = px.histogram(
                                df,
                                x='Hit_Probability',
                                nbins=20,
                                title='Hit Probability Distribution',
                                labels={'Hit_Probability': 'Probability of Being a Hit'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("❌ Model not loaded. Please train the model first.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Make sure your CSV has all required columns with correct names.")
    
    else:
        # Show example CSV format
        st.info("📁 Upload a CSV file to get started. Here's the required format:")
        
        example_df = pd.DataFrame({
            'track_name': ['Example Song 1', 'Example Song 2'],
            'artists': ['Artist A', 'Artist B'],
            'danceability': [0.8, 0.6],
            'energy': [0.9, 0.5],
            'valence': [0.7, 0.4],
            'tempo': [120, 95],
            'loudness': [-5, -8],
            'speechiness': [0.05, 0.08],
            'acousticness': [0.1, 0.6],
            'instrumentalness': [0.0, 0.2],
            'liveness': [0.2, 0.15],
            'artist_followers': [5000000, 100000],
            'artist_popularity': [85, 55],
            'lastfm_playcount': [1000000, 50000],
            'lastfm_listeners': [100000, 10000]
        })
        
        st.dataframe(example_df)
        
        # Download example
        csv = example_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Example CSV Template",
            data=csv,
            file_name="example_songs.csv",
            mime="text/csv"
        )

#═══════════════════════════════════════════════════════════
# PAGE 4: MODEL INFO
#═══════════════════════════════════════════════════════════
elif page == "📈 Model Info":
    st.header("📈 Model Information")
    
    if metrics is None:
        st.error("❌ Model metrics not available. Please train the model in your Jupyter notebook first.")
        st.info("""
        **To generate metrics:**
        1. Open your Jupyter notebook
        2. Run the model training code
        3. Add the metrics saving code at the end
        4. Refresh this Streamlit app
        """)
        st.stop()
    
    # Extract metrics
    accuracy = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    f1_score = metrics['f1_score']
    cm = metrics['confusion_matrix']
    feature_importance_df = metrics['feature_importance']
    
    # Model performance - NOW WITH REAL DATA!
    st.subheader("🎯 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Accuracy", f"{accuracy:.1%}")
    col2.metric("Precision", f"{precision:.1%}")
    col3.metric("Recall", f"{recall:.1%}")
    col4.metric("F1-Score", f"{f1_score:.1%}")
    
    st.info(f"📊 Evaluated on {metrics['test_size']} test songs with {metrics['num_features']} features")
    
    # Confusion matrix - FROM ACTUAL TEST DATA
    st.markdown("---")
    st.subheader("📊 Confusion Matrix")
    
    # Extract values from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**True Negatives (TN):** {tn} - Correctly predicted 'Not Hit'")
        st.write(f"**False Positives (FP):** {fp} - Incorrectly predicted 'Hit'")
    with col2:
        st.write(f"**False Negatives (FN):** {fn} - Incorrectly predicted 'Not Hit'")
        st.write(f"**True Positives (TP):** {tp} - Correctly predicted 'Hit'")
    
    confusion_data = pd.DataFrame({
        'Predicted Not Hit': [tn, fn],
        'Predicted Hit': [fp, tp]
    }, index=['Actual Not Hit', 'Actual Hit'])
    
    fig = px.imshow(
        confusion_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Greens',
        title="Random Forest - Confusion Matrix (Test Set Results)",
        labels=dict(x="Predicted Label", y="True Label", color="Count")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance - FROM ACTUAL MODEL
    st.markdown("---")
    st.subheader("🔍 Feature Importance")
    
    st.write("The model considers these features in order of importance:")
    
    # Get top 10 features
    top_features = feature_importance_df.head(10)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Most Important Features',
        color='importance',
        color_continuous_scale='Viridis',
        labels={'importance': 'Importance Score', 'feature': 'Feature'}
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show full feature importance table
    with st.expander("📋 View All Feature Importances"):
        st.dataframe(
            feature_importance_df.style.format({'importance': '{:.4f}'}),
            use_container_width=True
        )
    
    # Key insights based on actual data
    st.markdown("---")
    st.subheader("💡 Key Insights")
    
    top_feature = feature_importance_df.iloc[0]
    second_feature = feature_importance_df.iloc[1]
    third_feature = feature_importance_df.iloc[2]
    
    st.write(f"""
    **What the model learned from the data:**
    
    1. **{top_feature['feature'].replace('_', ' ').title()} ({top_feature['importance']:.1%})**
       - This is the most influential factor in predicting hits
       - The model weights this feature most heavily in its decisions
    
    2. **{second_feature['feature'].replace('_', ' ').title()} ({second_feature['importance']:.1%})**
       - Second most important predictor
       - Strong correlation with hit success
    
    3. **{third_feature['feature'].replace('_', ' ').title()} ({third_feature['importance']:.1%})**
       - Third key factor in the model's decisions
       - Helps distinguish between hits and non-hits
    
    **Model Performance Summary:**
    - ✅ Correctly identified **{tp}** out of **{tp + fn}** actual hits (**{recall:.1%}** recall)
    - ✅ **{precision:.1%}** of predicted hits were actually hits (precision)
    - ✅ Overall accuracy: **{accuracy:.1%}** on test data
    - ⚠️ Missed **{fn}** actual hits (false negatives)
    - ⚠️ Incorrectly predicted **{fp}** non-hits as hits (false positives)
    """)
    
    # Additional insights
    st.markdown("### 🎯 Practical Takeaways")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ Model Strengths:**
        - High accuracy in identifying hits
        - Balances precision and recall well
        - Considers multiple signal types
        - Trained on diverse genres
        """)
    
    with col2:
        st.warning("""
        **⚠️ Limitations:**
        - Requires complete feature data
        - Performance depends on data quality
        - May not capture viral trends
        - Artist popularity heavily weighted
        """)
    
    # Model details
    st.markdown("---")
    st.subheader("⚙️ Technical Details")
    
    st.write(f"""
    **Model:** Random Forest Classifier
    
    **Parameters:**
    - Number of trees: 100
    - Max depth: 10
    - Training samples: {metrics['test_size'] * 4} songs (80%)
    - Test samples: {metrics['test_size']} songs (20%)
    - Random state: 42 (reproducible)
    
    **Data Sources:**
    - Kaggle Spotify Dataset
    - Spotify Web API
    - Last.fm API
    
    **Total Features:** {metrics['num_features']}
    - Audio features: 9 (danceability, energy, valence, etc.)
    - Artist metrics: 2 (followers, popularity)
    - Engagement data: 2 (Last.fm plays, listeners)
    - Engineered features: 4 (quality index, visibility score, etc.)
    
    **Training Date:** December 2025
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | Data Science Project 2025</p>
    <p>🎵 Music Hit Classifier | Northeastern University</p>
</div>
""", unsafe_allow_html=True)