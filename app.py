"""
AI Performance Report - Streamlit Dashboard
==========================================
Interactive dashboard for analyzing AI chatbot performance with proper statistical analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
from collections import Counter
import re

warnings.filterwarnings('ignore')

# Load intent mapping
try:
    with open('intent_mapping.json', 'r') as f:
        INTENT_MAPPING = json.load(f)
except FileNotFoundError:
    INTENT_MAPPING = {}
    st.warning("⚠️ intent_mapping.json not found. Showing intent codes instead of descriptions.")

def get_intent_display_name(intent_code):
    """Get display name for intent code, fallback to code if not found"""
    return INTENT_MAPPING.get(intent_code, intent_code)

# Optional imports for topic modeling
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AI Performance Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Intent categorization
INTENT_CATEGORIES = {
    'Actionable': ['SS', 'STD', 'VYT', 'SYC', 'BV', 'GPQ', 'MDS', 'SC', 'OPT', 'FV', 'DS', 'TXT'],
    'FAQ': ['FAQ', 'faq'],
    'General': ['GSL', 'GSR'],
    'Irrelevant': ['IR', 'ir']
}

# Reverse mapping for quick lookup
INTENT_TO_CATEGORY = {}
for category, intents in INTENT_CATEGORIES.items():
    for intent in intents:
        INTENT_TO_CATEGORY[intent] = category


class AIPerformanceAnalyzer:
    """Analyzes AI chatbot performance with focus on accuracy and reliability"""
    
    def __init__(self, df):
        """Initialize with DataFrame"""
        self.df = df.copy()
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess and clean data"""
        # Parse JSON responses
        self.df['parsed_response'] = self.df['aiResponse_json'].apply(self._parse_json)
        self.df['dateCreatedUtc'] = pd.to_datetime(self.df['dateCreatedUtc'])
        
        # Convert UTC to EST
        # Check if already timezone-aware
        if self.df['dateCreatedUtc'].dt.tz is None:
            # If naive, localize to UTC first
            self.df['dateCreatedUtc'] = self.df['dateCreatedUtc'].dt.tz_localize('UTC')
        
        # Convert to Eastern Time
        self.df['dateCreatedEst'] = self.df['dateCreatedUtc'].dt.tz_convert('US/Eastern')
        
        # Extract features
        self.df['intentDescription'] = self.df['parsed_response'].apply(
            lambda x: x.get('intentDescription', '')
        )
        self.df['has_vehicle_info'] = self.df['parsed_response'].apply(
            lambda x: bool(x.get('interestedMake') or x.get('interestedModel'))
        )
        self.df['entity_count'] = self.df['parsed_response'].apply(self._count_entities)
        
        # Categorize intents
        self.df['intent_category'] = self.df['intentName'].apply(
            lambda x: INTENT_TO_CATEGORY.get(x, 'Unknown')
        )
        self.df['is_actionable'] = self.df['intent_category'] == 'Actionable'
        
        # Filter for Total Interactions (exclude IR+initial and IR+FAQ combinations)
        # Check if promptName column exists
        if 'promptName' in self.df.columns:
            self.df['is_valid_interaction'] = ~(
                (self.df['intentName'].str.upper().str.contains('IR', na=False)) & 
                (
                    self.df['promptName'].str.lower().str.contains('initial', na=False) | 
                    self.df['promptName'].str.lower().str.contains('faq', na=False)
                )
            )
        else:
            # If promptName doesn't exist, all rows are valid interactions
            self.df['is_valid_interaction'] = True
        
        # Time features (using EST time)
        self.df['hour'] = self.df['dateCreatedEst'].dt.hour
        self.df['day_of_week'] = self.df['dateCreatedEst'].dt.dayofweek
        self.df['date'] = self.df['dateCreatedEst'].dt.date
        
    @staticmethod
    def _parse_json(json_str):
        """Safely parse JSON string"""
        try:
            if pd.isna(json_str):
                return {}
            return json.loads(json_str.replace('""', '"'))
        except:
            return {}
    
    @staticmethod
    def _count_entities(parsed):
        """Count extracted entities"""
        exclude_keys = ['intentCode', 'intentDescription', 'aiResponse']
        return sum(1 for k, v in parsed.items() if k not in exclude_keys and v)
    
    def get_basic_metrics(self):
        """Calculate basic KPIs"""
        # Filter dataframe for valid interactions (excluding IR+initial and IR+FAQ)
        valid_df = self.df[self.df['is_valid_interaction']]
        
        metrics = {
            'total_ai_calls': len(self.df),  # All rows for cost calculation
            'total_interactions': len(valid_df),  # Filtered interactions for other metrics
            'unique_sessions': valid_df['sessionId'].nunique(),
            'unique_dealers': valid_df['dealerId'].nunique(),
            'unique_intents': valid_df['intentName'].nunique(),
            'actionable_rate': valid_df['is_actionable'].mean() * 100,
            'avg_session_length': valid_df.groupby('sessionId').size().mean(),
            'date_range': {
                'start': self.df['dateCreatedEst'].min().strftime('%Y-%m-%d %H:%M EST'),
                'end': self.df['dateCreatedEst'].max().strftime('%Y-%m-%d %H:%M EST')
            }
        }
        
        # Add cost metrics if available (use total_ai_calls data)
        if 'pricePerCall' in self.df.columns:
            metrics['total_cost'] = self.df['pricePerCall'].sum()
        
        # Add token metrics if available (use total_ai_calls data)
        if 'inputTokenCount' in self.df.columns and 'outputTokenCount' in self.df.columns:
            metrics['total_input_tokens'] = self.df['inputTokenCount'].sum()
            metrics['total_output_tokens'] = self.df['outputTokenCount'].sum()
            metrics['total_tokens'] = metrics['total_input_tokens'] + metrics['total_output_tokens']
        
        return metrics
    
    def get_intent_analysis(self):
        """Analyze intent distribution"""
        # Use valid interactions for analysis
        valid_df = self.df[self.df['is_valid_interaction']]
        
        # Overall intent stats
        intent_stats = valid_df.groupby('intentName').agg({
            'sessionId': 'count'
        }).round(2)
        
        intent_stats.columns = ['count']
        intent_stats = intent_stats.sort_values('count', ascending=False)
        intent_stats['percentage'] = (intent_stats['count'] / len(valid_df) * 100).round(2)
        
        # Map intent codes to descriptions
        intent_stats.index = intent_stats.index.map(get_intent_display_name)
        
        # Category-level stats
        category_stats = valid_df.groupby('intent_category').agg({
            'sessionId': 'count'
        }).round(2)
        
        category_stats.columns = ['count']
        category_stats['percentage'] = (category_stats['count'] / len(valid_df) * 100).round(2)
        
        return intent_stats, category_stats
    
    def get_entity_extraction_analysis(self):
        """Analyze entity extraction at intent level with information categories"""
        
        # Use valid interactions for analysis
        valid_df = self.df[self.df['is_valid_interaction']]
        
        # Intent-level extraction with multiple information categories
        intent_summary = []
        for intent in valid_df['intentName'].unique():
            intent_df = valid_df[valid_df['intentName'] == intent]
            if len(intent_df) > 0:
                # Check for different types of information
                current_vehicle_count = 0
                interested_vehicle_count = 0
                personal_info_count = 0
                appointments_count = 0
                service_info_count = 0
                finance_info_count = 0
                
                for _, row in intent_df.iterrows():
                    parsed = row['parsed_response']
                    
                    # Current Vehicle Info
                    if any(parsed.get(k) for k in ['currentYear', 'currentMake', 'currentModel', 'currentTrim']):
                        current_vehicle_count += 1
                    
                    # Interested Vehicle Info
                    if any(parsed.get(k) for k in ['interestedYear', 'interestedMake', 'interestedModel', 'interestedTrim', 'color', 'interiorColor', 'bodyType']):
                        interested_vehicle_count += 1
                    
                    # Personal Info
                    if any(parsed.get(k) for k in ['firstName', 'lastName', 'middleName', 'email', 'phoneNumber', 'address', 'zipCode', 'city', 'dateOfBirth']):
                        personal_info_count += 1
                    
                    # Appointments
                    if any(parsed.get(k) for k in ['appointmentDate', 'appointmentTime']):
                        appointments_count += 1
                    
                    # Service Info
                    if any(parsed.get(k) for k in ['serviceTypes', 'mileage']):
                        service_info_count += 1
                    
                    # Finance Info
                    if any(parsed.get(k) for k in ['downPayment', 'mileageAllowance', 'term', 'financeType']):
                        finance_info_count += 1
                
                intent_summary.append({
                    'intent': intent,
                    'total': len(intent_df),
                    'current_vehicle': current_vehicle_count,
                    'current_vehicle_rate': round(current_vehicle_count / len(intent_df) * 100, 1),
                    'interested_vehicle': interested_vehicle_count,
                    'interested_vehicle_rate': round(interested_vehicle_count / len(intent_df) * 100, 1),
                    'personal_info': personal_info_count,
                    'personal_info_rate': round(personal_info_count / len(intent_df) * 100, 1),
                    'appointments': appointments_count,
                    'appointments_rate': round(appointments_count / len(intent_df) * 100, 1),
                    'service_info': service_info_count,
                    'service_info_rate': round(service_info_count / len(intent_df) * 100, 1),
                    'finance_info': finance_info_count,
                    'finance_info_rate': round(finance_info_count / len(intent_df) * 100, 1)
                })
        
        intent_extraction = pd.DataFrame(intent_summary).sort_values('total', ascending=False)
        intent_extraction = intent_extraction.set_index('intent')
        
        # Map intent codes to descriptions
        intent_extraction.index = intent_extraction.index.map(get_intent_display_name)
        
        return intent_extraction
    

    def get_dealer_comparison(self):
        """Compare dealer performance"""
        # Use valid interactions for analysis
        valid_df = self.df[self.df['is_valid_interaction']]
        
        dealer_stats = valid_df.groupby('dealerId').agg({
            'sessionId': ['count', 'nunique'],
            'is_actionable': 'mean'
        }).round(2)
        
        dealer_stats.columns = ['interactions', 'sessions', 'actionable_rate']
        dealer_stats['msgs_per_session'] = (
            dealer_stats['interactions'] / dealer_stats['sessions']
        ).round(2)
        dealer_stats['actionable_rate'] = (dealer_stats['actionable_rate'] * 100).round(2)
        
        # Calculate category rates for each dealer
        for dealer_id in dealer_stats.index:
            dealer_df = valid_df[valid_df['dealerId'] == dealer_id]
            total = len(dealer_df)
            
            general_count = len(dealer_df[dealer_df['intent_category'] == 'General'])
            faq_count = len(dealer_df[dealer_df['intent_category'] == 'FAQ'])
            irrelevant_count = len(dealer_df[dealer_df['intent_category'] == 'Irrelevant'])
            
            dealer_stats.loc[dealer_id, 'general_rate'] = round(general_count / total * 100, 2)
            dealer_stats.loc[dealer_id, 'faq_rate'] = round(faq_count / total * 100, 2)
            dealer_stats.loc[dealer_id, 'irrelevant_rate'] = round(irrelevant_count / total * 100, 2)
        
        return dealer_stats
    
    def get_dealer_cost_analysis(self):
        """Analyze cost and token usage per dealer"""
        # Check if cost/token columns exist
        cost_cols = ['pricePerCall', 'inputTokenCount', 'outputTokenCount', 'latencyMs']
        available_cols = [col for col in cost_cols if col in self.df.columns]
        
        if not available_cols:
            return None
        
        # Aggregate cost and token metrics by dealer (sum only)
        agg_dict = {}
        if 'pricePerCall' in self.df.columns:
            agg_dict['pricePerCall'] = 'sum'
        if 'inputTokenCount' in self.df.columns:
            agg_dict['inputTokenCount'] = 'sum'
        if 'outputTokenCount' in self.df.columns:
            agg_dict['outputTokenCount'] = 'sum'
        
        dealer_costs = self.df.groupby('dealerId').agg(agg_dict).round(4)
        
        # Rename columns
        new_cols = []
        for col in dealer_costs.columns:
            if 'price' in col.lower():
                new_cols.append('total_price')
            elif 'inputToken' in col:
                new_cols.append('total_input_tokens')
            elif 'outputToken' in col:
                new_cols.append('total_output_tokens')
            else:
                new_cols.append(f'total_{col}')
        
        dealer_costs.columns = new_cols
        
        # Add AI calls count for cost context (use all data)
        dealer_costs['total_ai_calls'] = self.df.groupby('dealerId').size()
        
        # Calculate total tokens if both input and output available
        if 'total_input_tokens' in dealer_costs.columns and 'total_output_tokens' in dealer_costs.columns:
            dealer_costs['total_all_tokens'] = (
                dealer_costs['total_input_tokens'] + dealer_costs['total_output_tokens']
            ).astype(int)
        
        # Reorder columns for better readability
        col_order = ['total_ai_calls']
        if 'total_price' in dealer_costs.columns:
            col_order.append('total_price')
        if 'total_input_tokens' in dealer_costs.columns:
            col_order.append('total_input_tokens')
        if 'total_output_tokens' in dealer_costs.columns:
            col_order.append('total_output_tokens')
        if 'total_all_tokens' in dealer_costs.columns:
            col_order.append('total_all_tokens')
        
        # Keep only columns that exist
        col_order = [col for col in col_order if col in dealer_costs.columns]
        dealer_costs = dealer_costs[col_order]
        
        return dealer_costs
    
    def get_temporal_patterns(self):
        """Analyze temporal patterns"""
        # Use valid interactions for temporal patterns
        valid_df = self.df[self.df['is_valid_interaction']]
        
        return {
            'by_hour': valid_df.groupby('hour').size().to_dict(),
            'by_day': valid_df.groupby('day_of_week').size().to_dict(),
            'by_date': valid_df.groupby('date').size().to_dict()
        }
    
    @st.cache_resource
    def perform_topic_modeling(_self, intent_name, min_docs=10):
        """
        Perform enhanced BERTopic analysis on user messages for a specific intent
        
        Improvements for better accuracy:
        - Enhanced text preprocessing (removes URLs, emails, phones, special chars)
        - Better embedding model (all-mpnet-base-v2 for superior semantic understanding)
        - Custom vectorizer with n-grams (1-3 words) for better phrase detection
        - Domain-specific stopword removal for clearer topic words
        - Dynamic HDBSCAN clustering optimized for dataset size
        - Automatic topic reduction to prevent over-clustering
        - Updated c-TF-IDF representation for more coherent topics
        
        Args:
            intent_name: The intent to analyze (GSL, GSR, IR, FAQ)
            min_docs: Minimum number of documents required for analysis
            
        Returns:
            Dictionary with topic_model, topics, topic_info, and all_docs (ALL messages per topic)
        """
        if not BERTOPIC_AVAILABLE:
            return None
        
        # Check if userMessage column exists
        if 'userMessage' not in _self.df.columns:
            return {
                'error': 'userMessage column not found in dataset.',
                'count': 0
            }
        
        # Use filtered valid interactions (excluding IR+initial and IR+FAQ)
        valid_df = _self.df[_self.df['is_valid_interaction']]
        
        # Filter data for the specific intent (case-insensitive)
        intent_df = valid_df[valid_df['intentName'].str.upper() == intent_name.upper()].copy()
        
        if len(intent_df) < min_docs:
            return {
                'error': f'Insufficient data: Only {len(intent_df)} messages found. Need at least {min_docs}.',
                'count': len(intent_df)
            }
        
        # Extract and clean user messages with enhanced preprocessing
        docs = intent_df['userMessage'].fillna('').astype(str).tolist()
        
        # Enhanced cleaning: Remove URLs, emails, phone numbers, special chars
        docs_cleaned = []
        for doc in docs:
            # Remove URLs
            doc = re.sub(r"http\S+|www\S+", "", doc)
            # Remove emails
            doc = re.sub(r'\S+@\S+', "", doc)
            # Remove phone patterns
            doc = re.sub(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', "", doc)
            # Remove excessive punctuation but keep sentence structure
            doc = re.sub(r'[^\w\s\.\,\?\!]', ' ', doc)
            # Remove numbers (unless you want to keep them)
            doc = re.sub(r'\b\d+\b', '', doc)
            # Lowercase and normalize whitespace
            doc = re.sub(r'\s+', ' ', doc.lower()).strip()
            
            if len(doc) > 3:  # Filter out very short texts
                docs_cleaned.append(doc)
            else:
                docs_cleaned.append("empty_message")
        
        if len(docs_cleaned) < min_docs:
            return {
                'error': f'Insufficient valid messages after cleaning.',
                'count': len(docs_cleaned)
            }
        
        try:
            # Use better embedding model for improved semantic understanding
            # Options: 'all-MiniLM-L6-v2' (faster) or 'all-mpnet-base-v2' (better quality)
            embedding_model = SentenceTransformer('all-mpnet-base-v2')
            
            # Configure vectorizer with custom stopwords for better topic words
            from sklearn.feature_extraction.text import CountVectorizer
            
            # Domain-specific stopwords for automotive/chatbot context
            custom_stopwords = [
                'hi', 'hello', 'hey', 'thanks', 'thank', 'please', 'yes', 'no',
                'ok', 'okay', 'sure', 'want', 'need', 'like', 'know', 'get',
                'looking', 'help', 'find', 'show', 'tell', 'give', 'got', 'just',
                'can', 'could', 'would', 'will', 'may', 'might', 'should',
                'also', 'really', 'like', 'much', 'many', 'well', 'good', 'great'
            ]
            
            vectorizer_model = CountVectorizer(
                ngram_range=(1, 3),  # Capture 1-3 word phrases
                stop_words='english',  # Remove common English stopwords
                min_df=1,  # Minimum document frequency
                max_df=0.95  # Remove words appearing in >95% of docs
            )
            
            # Configure HDBSCAN for small datasets
            try:
                from hdbscan import HDBSCAN
                # Adjust parameters based on dataset size
                dataset_size = len(docs_cleaned)
                min_cluster_size = max(2, min(5, dataset_size // 10))
                min_samples = max(1, min(3, dataset_size // 20))
                
                hdbscan_model = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='euclidean',
                    cluster_selection_method='eom',
                    prediction_data=False
                )
                
                # Create BERTopic model with all enhancements
                topic_model = BERTopic(
                    embedding_model=embedding_model,
                    hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer_model,
                    language='english',
                    nr_topics='auto',
                    calculate_probabilities=False,
                    verbose=False,
                    low_memory=False,  # Better quality representations
                    min_topic_size=max(2, dataset_size // 20)  # Dynamic min topic size
                )
            except ImportError:
                # Fallback to KMeans if HDBSCAN not available
                from sklearn.cluster import KMeans
                n_topics = min(8, max(3, len(docs_cleaned) // 15))
                kmeans_model = KMeans(n_clusters=n_topics, random_state=42)
                
                topic_model = BERTopic(
                    embedding_model=embedding_model,
                    hdbscan_model=kmeans_model,
                    vectorizer_model=vectorizer_model,
                    language='english',
                    calculate_probabilities=False,
                    verbose=False,
                    min_topic_size=max(2, dataset_size // 20)
                )
            
            # Fit and transform
            topics, probs = topic_model.fit_transform(docs_cleaned)
            
            # Automatic topic reduction for better coherence
            # If too many topics are found, reduce them
            topic_info_initial = topic_model.get_topic_info()
            num_topics_found = len(topic_info_initial) - 1  # Exclude outliers
            
            # Reduce topics if we have too many for small datasets
            if num_topics_found > max(5, dataset_size // 15):
                target_topics = max(3, min(5, dataset_size // 15))
                topic_model.reduce_topics(docs_cleaned, nr_topics=target_topics)
                topics = topic_model.topics_
            
            # Get final topic information
            topic_info = topic_model.get_topic_info()
            
            # Update topic representation with c-TF-IDF improvements
            topic_model.update_topics(
                docs_cleaned, 
                vectorizer_model=vectorizer_model
            )
            
            # Get updated topic info
            topic_info = topic_model.get_topic_info()
            
            # Add topics to dataframe
            intent_df['topic'] = topics
            intent_df['cleaned_message'] = docs_cleaned
            
            # Get ALL documents for each topic (not just samples)
            all_docs = {}
            for topic_id in topic_info['Topic'].values:
                if topic_id != -1:  # Skip outliers
                    topic_docs = intent_df[intent_df['topic'] == topic_id]['userMessage'].tolist()
                    all_docs[topic_id] = topic_docs
            
            return {
                'topic_model': topic_model,
                'topics': topics,
                'topic_info': topic_info,
                'all_docs': all_docs,  # Changed from sample_docs to all_docs
                'intent_df': intent_df,
                'total_docs': len(docs_cleaned)
            }
            
        except Exception as e:
            return {
                'error': f'Error during topic modeling: {str(e)}',
                'count': len(docs_cleaned)
            }


def create_intent_pie_chart(intent_stats):
    """Create pie chart for individual intents"""
    
    fig = go.Figure(data=[go.Pie(
        labels=intent_stats.index,
        values=intent_stats['count'],
        hole=0.3,
        textinfo='label+percent',
        textposition='auto',
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='white', width=2)
        )
    )])
    
    fig.update_layout(
        title="Intent Distribution (All Intents)",
        height=500,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5)
    )
    
    return fig


def create_category_pie_chart(category_stats):
    """Create pie chart for intent categories"""
    colors = {
        'Actionable': '#2ecc71',
        'General': '#3498db',
        'FAQ': '#f39c12',
        'Irrelevant': '#e74c3c',
        'Unknown': '#95a5a6'
    }
    
    color_map = [colors.get(cat, '#95a5a6') for cat in category_stats.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=category_stats.index,
        values=category_stats['count'],
        hole=0.3,
        textinfo='label+percent+value',
        textposition='auto',
        marker=dict(
            colors=color_map,
            line=dict(color='white', width=3)
        )
    )])
    
    fig.update_layout(
        title="Intent Category Distribution",
        height=400,
        annotations=[dict(text='Categories', x=0.5, y=0.5, font_size=16, showarrow=False)]
    )
    
    return fig





def create_temporal_chart(temporal_data):
    """Create hourly pattern chart"""
    hour_data = temporal_data['by_hour']
    hours = sorted(hour_data.keys())
    counts = [hour_data[h] for h in hours]
    
    fig = go.Figure(data=[
        go.Scatter(
            x=hours,
            y=counts,
            mode='lines+markers',
            line=dict(color='#9b59b6', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(155, 89, 182, 0.2)'
        )
    ])
    
    fig.update_layout(
        title="Interaction Volume by Hour of Day (EST)",
        xaxis_title="Hour (EST)",
        yaxis_title="Number of Interactions",
        height=400,
        xaxis=dict(tickmode='linear', dtick=2)
    )
    
    return fig


def dealer_selection_page(df, dealer_details_df):
    """Dealer selection page with searchable table interface"""
    st.title("🏢 Dealer Selection")
    st.markdown("### Select dealers to analyze")
    
    # Get unique dealer IDs from pilot data
    available_dealers = set(df['dealerId'].unique())
    
    # Create dealer list with details
    dealer_list = []
    for dealer_id in sorted(available_dealers):
        dealer_info = dealer_details_df[dealer_details_df['DealerId'] == dealer_id]
        if not dealer_info.empty:
            dealer_list.append({
                'DealerId': dealer_id,
                'ClientName': dealer_info.iloc[0]['ClientName'],
                'Domain': dealer_info.iloc[0].get('Domain', 'N/A')
            })
        else:
            dealer_list.append({
                'DealerId': dealer_id,
                'ClientName': 'Unknown',
                'Domain': 'N/A'
            })
    
    dealer_df = pd.DataFrame(dealer_list)
    
    st.info(f"📊 Total Available Dealers: **{len(dealer_df)}**")
    
    # Initialize selected dealers in session state
    if 'selected_dealers' not in st.session_state:
        st.session_state.selected_dealers = []
    
    # Search filters - live search as you type
    col1, col2 = st.columns(2)
    
    with col1:
        search_dealer_id = st.text_input(
            "🔍 Search by Dealer ID",
            placeholder="Enter dealer ID...",
            key="search_dealer_id",
            label_visibility="visible"
        )
    
    with col2:
        search_client_name = st.text_input(
            "🔍 Search by Client Name",
            placeholder="Enter client name...",
            key="search_client_name",
            label_visibility="visible"
        )
    
    # Filter the dataframe based on search - updates live as you type
    filtered_df = dealer_df.copy()
    
    if search_dealer_id:
        filtered_df = filtered_df[
            filtered_df['DealerId'].astype(str).str.contains(search_dealer_id, case=False, na=False)
        ]
    
    if search_client_name:
        filtered_df = filtered_df[
            filtered_df['ClientName'].str.contains(search_client_name, case=False, na=False)
        ]
    
    st.markdown(f"**Showing {len(filtered_df)} dealer(s)**")
    
    # Selection options
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("✅ Select All", use_container_width=True):
            st.session_state.selected_dealers = filtered_df['DealerId'].tolist()
            st.session_state.selection_version += 1
            st.rerun()
    
    with col2:
        if st.button("❌ Clear All", use_container_width=True):
            st.session_state.selected_dealers = []
            st.session_state.selection_version += 1
            st.rerun()
    
    # Display dealer table with checkboxes
    st.markdown("---")
    
    # Create selection interface
    for idx, row in filtered_df.iterrows():
        dealer_id = row['DealerId']
        is_selected = dealer_id in st.session_state.selected_dealers
        
        col1, col2, col3 = st.columns([1, 3, 3])
        
        with col1:
            checked = st.checkbox("", value=is_selected, key=f"cb_{dealer_id}_v{st.session_state.selection_version}")
            if checked and dealer_id not in st.session_state.selected_dealers:
                st.session_state.selected_dealers.append(dealer_id)
            elif not checked and dealer_id in st.session_state.selected_dealers:
                st.session_state.selected_dealers.remove(dealer_id)
        
        with col2:
            st.markdown(f"**{row['ClientName']}**")
        
        with col3:
            st.markdown(f"`{dealer_id}`")
    
    # Bottom action bar
    st.markdown("---")
    
    # Show selected dealers count and names
    st.markdown(f"**Selected: {len(st.session_state.selected_dealers)} dealer(s)**")
    
    # Display selected dealer names
    if st.session_state.selected_dealers:
        st.markdown("**Selected Dealers:**")
        selected_dealer_info = []
        for dealer_id in st.session_state.selected_dealers:
            dealer_info = dealer_df[dealer_df['DealerId'] == dealer_id]
            if not dealer_info.empty:
                client_name = dealer_info.iloc[0]['ClientName']
                selected_dealer_info.append(f"• **{client_name}** (`{dealer_id}`)")
        
        # Display in columns for better layout
        if len(selected_dealer_info) > 0:
            st.markdown("\n".join(selected_dealer_info))
    
    st.markdown("---")
    
    # Action buttons
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📊 Analyze Selected", type="primary", use_container_width=True, disabled=len(st.session_state.selected_dealers) == 0):
            st.session_state.page = 'analysis'
            st.rerun()
    
    with col2:
        if st.button("📈 Analyze All Dealers", use_container_width=True):
            st.session_state.selected_dealers = dealer_df['DealerId'].tolist()
            st.session_state.page = 'analysis'
            st.rerun()


def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'selection'
    if 'selected_dealers' not in st.session_state:
        st.session_state.selected_dealers = []
    if 'selection_version' not in st.session_state:
        st.session_state.selection_version = 0
    
    # Sidebar for file upload (always visible)
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload your AI chatbot interaction data"
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file to begin analysis")
        st.markdown("""
        **Expected CSV columns:**
        - `sessionId`, `dealerId`, `intentName`, `userMessage`
        - `aiResponse_json`, `latencyMs`, `pricePerCall`
        - `inputTokenCount`, `outputTokenCount`, `dateCreatedUtc`
        """)
        return
    
    # Load data
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✅ Loaded {len(df)} records")
        
        # Load dealer details
        try:
            dealer_details_df = pd.read_csv('ProdDealerDetails.csv')
            dealer_details_available = True
        except FileNotFoundError:
            st.sidebar.warning("⚠️ ProdDealerDetails.csv not found. Analyzing all dealers.")
            dealer_details_available = False
            dealer_details_df = None
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return
    
    # Page navigation in sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("📄 Navigation")
    
    if dealer_details_available:
        if st.sidebar.button("🏢 Dealer Selection", use_container_width=True, type="primary" if st.session_state.page == 'selection' else "secondary"):
            st.session_state.page = 'selection'
            st.rerun()
        
        if st.sidebar.button("📊 Analysis Dashboard", use_container_width=True, disabled=len(st.session_state.selected_dealers) == 0, type="primary" if st.session_state.page == 'analysis' else "secondary"):
            st.session_state.page = 'analysis'
            st.rerun()
    
    # Route to appropriate page
    if dealer_details_available and st.session_state.page == 'selection':
        # Show dealer selection page
        dealer_selection_page(df, dealer_details_df)
        # Show count after processing selections
        if st.session_state.selected_dealers:
            st.sidebar.success(f"✅ {len(st.session_state.selected_dealers)} dealer(s) selected")
        return
    
    # Analysis page
    st.title("🤖 AI Performance Analytics Dashboard")
    st.markdown("### Comprehensive Analysis of AI Chatbot Performance")
    
    # Filter dataframe if dealers are selected
    if dealer_details_available and st.session_state.selected_dealers:
        df = df[df['dealerId'].isin(st.session_state.selected_dealers)]
        st.info(f"📊 Analyzing {len(st.session_state.selected_dealers)} selected dealer(s)")
        
        # Display selected dealers with their domains
        with st.expander("🏢 View Selected Dealers & Domains", expanded=False):
            dealers_info = []
            for dealer_id in sorted(st.session_state.selected_dealers):
                dealer_info = dealer_details_df[dealer_details_df['DealerId'] == dealer_id]
                if not dealer_info.empty:
                    domain = dealer_info.iloc[0].get('Domain', 'N/A')
                    # Add https:// if not present
                    if domain != 'N/A' and not domain.startswith('http'):
                        domain = f"https://{domain}"
                    dealers_info.append({
                        'Dealer ID': dealer_id,
                        'Client Name': dealer_info.iloc[0]['ClientName'],
                        'Domain': domain
                    })
                else:
                    dealers_info.append({
                        'Dealer ID': dealer_id,
                        'Client Name': 'Unknown',
                        'Domain': 'N/A'
                    })
            
            # Display as a nice table with clickable links
            dealers_display_df = pd.DataFrame(dealers_info)
            st.dataframe(
                dealers_display_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Domain": st.column_config.LinkColumn("Domain")
                }
            )
        
        if len(df) == 0:
            st.warning("⚠️ No data found for selected dealer(s)")
            return
    
    analyzer = AIPerformanceAnalyzer(df)
    
    # Section selector
    st.sidebar.header("📊 Report Sections")
    show_overview = st.sidebar.checkbox("Executive Overview", value=True)
    show_intent_analysis = st.sidebar.checkbox("Intent Analysis", value=True)
    show_category_analysis = st.sidebar.checkbox("Category Analysis", value=True)
    show_entity_extraction = st.sidebar.checkbox("Entity Extraction", value=True)
    show_dealer_comparison = st.sidebar.checkbox("Dealer Comparison", value=True)
    show_cost_analysis = st.sidebar.checkbox("Cost & Token Analysis", value=True)
    show_temporal = st.sidebar.checkbox("Temporal Patterns", value=True)
    
    if BERTOPIC_AVAILABLE:
        show_topic_modeling = st.sidebar.checkbox("Topic Modeling (GSL/GSR/IR/FAQ)", value=False)
    else:
        st.sidebar.warning("⚠️ Topic Modeling unavailable. Install: pip install bertopic sentence-transformers")
    
    # ==================== EXECUTIVE OVERVIEW ====================
    if show_overview:
        st.header("📊 Executive Overview")
        metrics = analyzer.get_basic_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total AI Calls", f"{metrics['total_ai_calls']:,}", help="Total number of AI API calls (all rows)")
        with col2:
            st.metric("Total Interactions", f"{metrics['total_interactions']:,}", help="Excluding IR+initial and IR+FAQ combinations")
        with col3:
            st.metric("Unique Sessions", f"{metrics['unique_sessions']:,}")
        with col4:
            st.metric("Actionable Rate", f"{metrics['actionable_rate']:.1f}%")
        
        st.markdown(f"""
        **Analysis Period:** {metrics['date_range']['start']} to {metrics['date_range']['end']}  
        **Dealers Analyzed:** {metrics['unique_dealers']}  
        **Intent Types:** {metrics['unique_intents']}
        """)
        
        # Cost and Token Metrics (if available)
        if 'total_cost' in metrics or 'total_tokens' in metrics:
            st.divider()
            st.subheader("💰 Cost & Token Overview")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'total_cost' in metrics:
                    st.metric("Total Cost", f"${metrics['total_cost']:.4f}")
                else:
                    st.metric("Total Cost", "N/A")
            
            with col2:
                if 'total_tokens' in metrics:
                    st.metric("Total Tokens", f"{int(metrics['total_tokens']):,}")
                else:
                    st.metric("Total Tokens", "N/A")
            
            # Additional detail row
            if 'total_input_tokens' in metrics and 'total_output_tokens' in metrics:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Input Tokens", f"{int(metrics['total_input_tokens']):,}")
                with col2:
                    st.metric("Output Tokens", f"{int(metrics['total_output_tokens']):,}")
        
        st.divider()
    
    # ==================== COST & TOKEN ANALYSIS ====================
    if show_cost_analysis:
        st.header("💰 Cost & Token Analysis by Dealer")
        
        dealer_costs = analyzer.get_dealer_cost_analysis()
        
        if dealer_costs is not None:
            # Detailed table
            st.subheader("📋 Detailed Cost & Token Breakdown by Dealer")
            
            # Create styled dataframe
            styled_df = dealer_costs.style
            
            if 'total_price' in dealer_costs.columns:
                styled_df = styled_df.background_gradient(subset=['total_price'], cmap='Reds')
            if 'total_all_tokens' in dealer_costs.columns:
                styled_df = styled_df.background_gradient(subset=['total_all_tokens'], cmap='Blues')
            if 'total_input_tokens' in dealer_costs.columns:
                styled_df = styled_df.background_gradient(subset=['total_input_tokens'], cmap='Greens')
            if 'total_output_tokens' in dealer_costs.columns:
                styled_df = styled_df.background_gradient(subset=['total_output_tokens'], cmap='Purples')
            
            # Format currency and numbers
            format_dict = {}
            for col in dealer_costs.columns:
                if 'price' in col.lower() or 'cost' in col.lower():
                    format_dict[col] = '${:.4f}'
            
            st.dataframe(styled_df, width='stretch')
        else:
            st.warning("⚠️ Cost and token data not available in the uploaded file.")
            st.info("💡 Expected columns: pricePerCall, inputTokenCount, outputTokenCount, latencyMs")
        
        st.divider()
    
    # ==================== CATEGORY ANALYSIS ====================
    if show_category_analysis:
        st.header("🎯 Intent Category Analysis")
        
        intent_stats, category_stats = analyzer.get_intent_analysis()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(create_category_pie_chart(category_stats), width='stretch')
        
        with col2:
            st.subheader("Category Statistics")
            st.dataframe(
                category_stats.style.background_gradient(subset=['count'], cmap='Blues'),
                width='stretch'
            )
        
        st.divider()
    
    # ==================== INTENT ANALYSIS ====================
    if show_intent_analysis:
        st.header("🎯 Detailed Intent Analysis")
        
        intent_stats, _ = analyzer.get_intent_analysis()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(create_intent_pie_chart(intent_stats), width='stretch')
        
        with col2:
            st.subheader("Top Intents Table")
            st.dataframe(
                intent_stats.head(5).style.background_gradient(subset=['count'], cmap='Greens'),
                width='stretch'
            )
        
        st.subheader("All Intent Statistics")
        st.dataframe(
            intent_stats.style.background_gradient(subset=['count'], cmap='Greens'),
            use_container_width=True
        )
        
        st.divider()
    
    # ==================== ENTITY EXTRACTION ====================
    if show_entity_extraction:
        st.header("🔍 Entity Extraction Performance")
        
        intent_extraction = analyzer.get_entity_extraction_analysis()
        
        st.subheader("Intent-Level Information Extraction")
        st.info("📊 **Categories Tracked**: Current Vehicle | Interested Vehicle | Personal Info | Appointments | Service Info | Finance Info")
        
        # Display the detailed table
        st.dataframe(
            intent_extraction.style.background_gradient(subset=['total'], cmap='Blues')
                                   .background_gradient(subset=['current_vehicle_rate'], cmap='Greens')
                                   .background_gradient(subset=['interested_vehicle_rate'], cmap='Oranges')
                                   .background_gradient(subset=['personal_info_rate'], cmap='Purples')
                                   .background_gradient(subset=['appointments_rate'], cmap='YlOrBr')
                                   .background_gradient(subset=['service_info_rate'], cmap='RdPu')
                                   .background_gradient(subset=['finance_info_rate'], cmap='YlGnBu'),
            width='stretch'
        )
        
        # Summary metrics
        st.subheader("Overall Summary")
        
        st.caption("📌 Note: Each metric shows extracted count vs relevant intent total (not all 317 records)")
        
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        
        # Current Vehicle Info: VYT, SYC, SS
        current_vehicle_intents = ['VYT', 'SYC', 'SS']
        current_vehicle_df = intent_extraction[intent_extraction.index.isin([get_intent_display_name(i) for i in current_vehicle_intents])]
        current_vehicle_total = current_vehicle_df['total'].sum()
        total_current_vehicle = current_vehicle_df['current_vehicle'].sum()
        
        # Interested Vehicle Info: BV, MDS, FV, STD
        interested_vehicle_intents = ['BV', 'MDS', 'FV', 'STD']
        interested_vehicle_df = intent_extraction[intent_extraction.index.isin([get_intent_display_name(i) for i in interested_vehicle_intents])]
        interested_vehicle_total = interested_vehicle_df['total'].sum()
        total_interested_vehicle = interested_vehicle_df['interested_vehicle'].sum()
        
        # Personal Info: All intents
        personal_total = intent_extraction['total'].sum()
        total_personal = intent_extraction['personal_info'].sum()
        
        # Appointments: SYC, STD, SS
        appointments_intents = ['SYC', 'STD', 'SS']
        appointments_df = intent_extraction[intent_extraction.index.isin([get_intent_display_name(i) for i in appointments_intents])]
        appointments_total = appointments_df['total'].sum()
        total_appointments = appointments_df['appointments'].sum()
        
        # Service Info: SS, SC
        service_intents = ['SS', 'SC']
        service_df = intent_extraction[intent_extraction.index.isin([get_intent_display_name(i) for i in service_intents])]
        service_total = service_df['total'].sum()
        total_service = service_df['service_info'].sum()
        
        # Finance Info: BV
        finance_intents = ['BV']
        finance_df = intent_extraction[intent_extraction.index.isin([get_intent_display_name(i) for i in finance_intents])]
        finance_total = finance_df['total'].sum()
        total_finance = finance_df['finance_info'].sum()
        
        with col1:
            if current_vehicle_total > 0:
                st.metric("Current Vehicle Info", 
                         f"{total_current_vehicle}/{current_vehicle_total}",
                         f"{total_current_vehicle/current_vehicle_total*100:.1f}%",
                         help=f"Relevant intents: {', '.join([get_intent_display_name(i) for i in current_vehicle_intents])}")
            else:
                st.metric("Current Vehicle Info", "0/0", "No relevant intents")
                
        with col2:
            if interested_vehicle_total > 0:
                st.metric("Interested Vehicle Info", 
                         f"{total_interested_vehicle}/{interested_vehicle_total}",
                         f"{total_interested_vehicle/interested_vehicle_total*100:.1f}%",
                         help=f"Relevant intents: {', '.join([get_intent_display_name(i) for i in interested_vehicle_intents])}")
            else:
                st.metric("Interested Vehicle Info", "0/0", "No relevant intents")
                
        with col3:
            if personal_total > 0:
                st.metric("Personal Info", 
                         f"{total_personal}/{personal_total}",
                         f"{total_personal/personal_total*100:.1f}%",
                         help="Relevant intents: All intents")
            else:
                st.metric("Personal Info", "0/0")
                
        with col4:
            if appointments_total > 0:
                st.metric("Appointments", 
                         f"{total_appointments}/{appointments_total}",
                         f"{total_appointments/appointments_total*100:.1f}%",
                         help=f"Relevant intents: {', '.join([get_intent_display_name(i) for i in appointments_intents])}")
            else:
                st.metric("Appointments", "0/0", "No relevant intents")
                
        with col5:
            if service_total > 0:
                st.metric("Service Info", 
                         f"{total_service}/{service_total}",
                         f"{total_service/service_total*100:.1f}%",
                         help=f"Relevant intents: {', '.join([get_intent_display_name(i) for i in service_intents])}")
            else:
                st.metric("Service Info", "0/0", "No relevant intents")
                
        with col6:
            if finance_total > 0:
                st.metric("Finance Info", 
                         f"{total_finance}/{finance_total}",
                         f"{total_finance/finance_total*100:.1f}%",
                         help=f"Relevant intents: {', '.join([get_intent_display_name(i) for i in finance_intents])}")
            else:
                st.metric("Finance Info", "0/0", "No relevant intents")
        
        st.divider()
    

    # ==================== DEALER COMPARISON ====================
    if show_dealer_comparison:
        st.header("🏢 Dealer Performance Comparison")
        
        dealer_stats = analyzer.get_dealer_comparison()
        
        st.dataframe(
            dealer_stats.style.background_gradient(subset=['interactions'], cmap='Blues')
                           .background_gradient(subset=['actionable_rate'], cmap='Greens')
                           .background_gradient(subset=['general_rate'], cmap='Purples')
                           .background_gradient(subset=['faq_rate'], cmap='Oranges')
                           .background_gradient(subset=['irrelevant_rate'], cmap='Reds'),
            width='stretch'
        )
        
        st.divider()
    
    # ==================== TEMPORAL PATTERNS ====================
    if show_temporal:
        st.header("⏰ Temporal Patterns")
        
        temporal_data = analyzer.get_temporal_patterns()
        
        st.plotly_chart(create_temporal_chart(temporal_data), width='stretch')
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_data = temporal_data['by_day']
        
        fig = px.bar(
            x=[day_names[i] for i in sorted(day_data.keys())],
            y=[day_data[i] for i in sorted(day_data.keys())],
            title="Interactions by Day of Week",
            labels={'x': 'Day', 'y': 'Count'},
            color=[day_data[i] for i in sorted(day_data.keys())],
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, width='stretch')
        
        st.divider()
    
    # ==================== TOPIC MODELING ====================
    if BERTOPIC_AVAILABLE and show_topic_modeling:
        st.header("🔬 Topic Modeling Analysis")
        st.markdown("**Analyzing user message patterns for GSL, GSR, IR, and FAQ intents**")
        st.info("💡 This analysis uses BERTopic to discover common themes and patterns in user queries for non-actionable intents.")
        
        # Show improvements in an expander
        with st.expander("🚀 Enhanced Accuracy Features", expanded=False):
            st.markdown("""
            **This analysis includes advanced techniques for better topic discovery:**
            
            1. **🧹 Enhanced Text Preprocessing**
               - Removes URLs, emails, phone numbers for cleaner analysis
               - Filters special characters while preserving sentence structure
               - Normalizes text for better pattern matching
            
            2. **🧠 Superior Embedding Model** 
               - Uses `all-mpnet-base-v2` (better than default)
               - Captures deeper semantic meaning and relationships
               - Improved understanding of chatbot-specific language
            
            3. **📝 Smart Phrase Detection**
               - N-gram analysis (1-3 words) captures multi-word phrases
               - Example: "schedule appointment" vs just "appointment"
               - Custom stopword removal for clearer topic keywords
            
            4. **🎯 Optimized Clustering**
               - Parameters dynamically adjusted based on dataset size
               - Automatic topic reduction prevents over-clustering
               - Better handling of small datasets (like your 40-50 messages)
            
            5. **📊 Improved Topic Representation**
               - Enhanced c-TF-IDF scoring for more meaningful topic words
               - Better topic coherence and interpretability
            """)
        
        # Create tabs for each intent
        tab1, tab2, tab3, tab4 = st.tabs([
            f"📊 {get_intent_display_name('GSL')}", 
            f"📊 {get_intent_display_name('GSR')}", 
            f"🚫 {get_intent_display_name('IR')}", 
            f"❓ {get_intent_display_name('FAQ')}"
        ])
        
        target_intents = {
            'GSL': tab1,
            'GSR': tab2,
            'IR': tab3,
            'FAQ': tab4
        }
        
        for intent_name, tab in target_intents.items():
            with tab:
                st.subheader(f"Topic Analysis: {get_intent_display_name(intent_name)}")
                
                with st.spinner(f"Analyzing {get_intent_display_name(intent_name)} messages..."):
                    result = analyzer.perform_topic_modeling(intent_name)
                
                if result is None:
                    st.error("❌ BERTopic libraries not available. Install with: `pip install bertopic sentence-transformers`")
                    continue
                
                if 'error' in result:
                    st.warning(f"⚠️ {result['error']}")
                    st.info(f"Found {result.get('count', 0)} messages for {get_intent_display_name(intent_name)}")
                    continue
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Messages", result['total_docs'])
                with col2:
                    num_topics = len(result['topic_info']) - 1  # Exclude outliers (-1)
                    st.metric("Topics Discovered", num_topics)
                with col3:
                    topics_array = np.array(result['topics'])
                    outliers = int(np.sum(topics_array == -1))
                    outlier_pct = (outliers / result['total_docs'] * 100) if result['total_docs'] > 0 else 0
                    st.metric("Outliers", f"{outliers} ({outlier_pct:.1f}%)")
                
                st.divider()
                
                # Topic Information Table
                st.subheader("📋 Discovered Topics")
                topic_info_display = result['topic_info'].copy()
                
                # Filter out outlier topic for display
                topic_info_display = topic_info_display[topic_info_display['Topic'] != -1]
                
                if len(topic_info_display) > 0:
                    # Rename columns for clarity
                    topic_info_display = topic_info_display.rename(columns={
                        'Topic': 'Topic ID',
                        'Count': 'Message Count',
                        'Name': 'Topic Keywords'
                    })
                    
                    st.dataframe(
                        topic_info_display[['Topic ID', 'Message Count', 'Topic Keywords']]
                        .style.background_gradient(subset=['Message Count'], cmap='Blues'),
                        width='stretch',
                        height=400
                    )
                else:
                    st.info("No clear topics discovered. Messages may be too diverse or too few.")
                
                st.divider()
                
                # Visualizations
                st.subheader("📊 Topic Visualizations")
                
                if num_topics > 0:
                    try:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Topic distribution bar chart
                            fig_bars = result['topic_model'].visualize_barchart(top_n_topics=min(8, num_topics))
                            st.plotly_chart(fig_bars, width='stretch')
                        
                        with col2:
                            # Topic distribution (excluding outliers)
                            topic_counts = result['intent_df'][result['intent_df']['topic'] != -1]['topic'].value_counts()
                            if len(topic_counts) > 0:
                                fig_pie = go.Figure(data=[go.Pie(
                                    labels=[f"Topic {t}" for t in topic_counts.index],
                                    values=topic_counts.values,
                                    hole=0.3
                                )])
                                fig_pie.update_layout(
                                    title=f"{get_intent_display_name(intent_name)} Topic Distribution",
                                    height=400
                                )
                                st.plotly_chart(fig_pie, width='stretch')
                    
                    except Exception as e:
                        st.warning(f"Could not generate all visualizations: {str(e)}")
                    
                    # Intertopic Distance Map
                    if num_topics >= 3:
                        try:
                            st.subheader("🗺️ Intertopic Distance Map")
                            fig_topics = result['topic_model'].visualize_topics()
                            st.plotly_chart(fig_topics, width='stretch')
                        except:
                            st.info("Topic distance map unavailable")
                    else:
                        st.info("🗺️ Topic distance map unavailable (need 3+ topics)")
                else:
                    st.info("No topics to visualize. All messages classified as outliers.")
                
                st.divider()
                
                # ALL messages per topic - Show everything for verification
                st.subheader("💬 All Messages by Topic (Complete View)")
                
                # Verification: Count total messages analyzed
                total_analyzed = sum(len(docs) for docs in result['all_docs'].values())
                total_outliers = outliers
                st.info(f"📊 **Verification:** {total_analyzed} messages in topics + {total_outliers} outliers = {total_analyzed + total_outliers} total messages analyzed")
                
                # Quick summary table
                st.markdown("**📋 Topic Summary:**")
                summary_data = []
                for topic_id in sorted(result['all_docs'].keys()):
                    topic_keywords = result['topic_info'][result['topic_info']['Topic'] == topic_id]['Name'].values
                    if len(topic_keywords) > 0:
                        msg_count = len(result['all_docs'][topic_id])
                        # Get top 5 keywords only
                        keywords_short = ', '.join(topic_keywords[0].split('_')[:5])
                        summary_data.append({
                            'Topic ID': topic_id,
                            'Message Count': msg_count,
                            'Top Keywords': keywords_short
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(
                        summary_df.style.background_gradient(subset=['Message Count'], cmap='Blues'),
                        width='stretch'
                    )
                
                st.markdown("---")
                st.markdown("**📝 Click on each topic below to see ALL messages:**")
                
                # Show ALL topics and ALL messages
                for topic_id in sorted(result['all_docs'].keys()):  # Show ALL topics
                    topic_keywords = result['topic_info'][result['topic_info']['Topic'] == topic_id]['Name'].values
                    if len(topic_keywords) > 0:
                        topic_messages = result['all_docs'][topic_id]
                        msg_count = len(topic_messages)
                        
                        with st.expander(f"🔸 Topic {topic_id}: {topic_keywords[0][:70]}... ({msg_count} messages)", expanded=False):
                            st.caption(f"**All {msg_count} messages in this topic:**")
                            for i, msg in enumerate(topic_messages, 1):
                                st.markdown(f"**{i}.** {msg}")
                
                # Show outlier messages too
                if total_outliers > 0:
                    outlier_messages = result['intent_df'][result['intent_df']['topic'] == -1]['userMessage'].tolist()
                    with st.expander(f"⚠️ Outlier Messages ({total_outliers} messages)", expanded=False):
                        st.caption(f"**Messages that didn't fit into any topic:**")
                        for i, msg in enumerate(outlier_messages, 1):
                            st.markdown(f"**{i}.** {msg}")
        
        st.divider()
    
    # Footer
    st.divider()
    st.markdown(f"""
    <div style='text-align: center; color: gray;'>
        Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
        Powered by Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
