#!/usr/bin/env python3
"""
KLM Operations Hub - Consolidated dashboard with all visualizations.
Enhanced with file upload and data cleaning capabilities.
Users can now upload raw CSV files and automatically clean them before analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io
import zipfile
import time
from datetime import datetime, timedelta
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

# Import DataCleaner and DataTypeChecker from streamlit_data_cleaner
try:
    from streamlit_data_cleaner import DataCleaner, DataTypeChecker
except ImportError:
    # Fallback: define basic cleaning functionality if import fails
    class DataCleaner:
        def __init__(self):
            self.original_data = {}

        def clean_dataframe(self, filename, df):
            # Basic cleaning implementation
            df_cleaned = df.copy()
            cleaning_log = []

            # Remove completely NULL rows (>90% NULL)
            total_columns = len(df_cleaned.columns)
            null_percentage_per_row = df_cleaned.isnull().sum(axis=1) / total_columns
            completely_null_rows = df_cleaned[null_percentage_per_row > 0.9].index

            if len(completely_null_rows) > 0:
                df_cleaned = df_cleaned.drop(completely_null_rows)
                cleaning_log.append(f"Removed {len(completely_null_rows)} completely NULL rows")

            # Remove very high NULL columns (>70%)
            total_rows = len(df_cleaned)
            columns_with_very_high_nulls = []

            for col in df_cleaned.columns:
                null_percentage = (df_cleaned[col].isnull().sum() / total_rows) * 100
                if null_percentage >= 70:
                    columns_with_very_high_nulls.append(col)
                    cleaning_log.append(f"Removed column '{col}' due to {null_percentage:.1f}% null values")

            if columns_with_very_high_nulls:
                df_cleaned = df_cleaned.drop(columns=columns_with_very_high_nulls)

            # Convert datetime columns
            for col in df_cleaned.columns:
                if any(word in col.lower() for word in ['time', 'date', 'created_on', 'updated_on']):
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')

            return df_cleaned, cleaning_log

        def analyze_data_quality(self, filename, df):
            return {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'null_values': df.isnull().sum().sum(),
                'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'empty_strings': (df == '').sum().sum(),
                'na_values': df.isna().sum().sum(),
                'columns_with_issues': []
            }

    class DataTypeChecker:
        def analyze_column_types(self, filename, df):
            suggestions = {}
            for col in df.columns:
                suggestions[col] = {
                    'current_type': str(df[col].dtype),
                    'suggested_type': 'object',
                    'notes': ['Basic analysis only']
                }
            return suggestions

# Set page configuration
st.set_page_config(
    page_title="KLM Visual Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set plotly configuration after imports
import plotly.io as pio
pio.templates.default = "plotly_white"

# Custom CSS (exact origineel)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066CC;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #0066CC;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-container {
        background: linear-gradient(135deg, #0066CC, #004499);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .explanation-box {
        background: linear-gradient(135deg, #f8fafc, #e2e8f0);
        border-left: 4px solid #0066CC;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        font-size: 0.9rem;
        color: #334155;
    }
    .stats-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.25rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .stats-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0066CC;
    }
    .stats-label {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

class KLMDataProcessor:
    """Load and process KLM data for visualization with support for uploaded files"""

    def __init__(self, data_path="cleaned_data", use_uploaded_data=False):
        if not os.path.isabs(data_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_path = os.path.join(current_dir, data_path)
        else:
            self.data_path = data_path
        self.data = {}
        self.use_uploaded_data = use_uploaded_data

    def load_all_data(self):
        """Load all CSV files with proper datetime handling - from uploaded data or files"""
        if self.use_uploaded_data and 'cleaned_data' in st.session_state:
            # Use uploaded and cleaned data
            self.data = st.session_state.cleaned_data.copy()
            print("✓ Using uploaded cleaned data")

            # Ensure datetime columns are properly converted
            for key, df in self.data.items():
                if not df.empty:
                    for col in df.columns:
                        if any(word in col.lower() for word in ['time', 'date', 'created_on', 'updated_on']):
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                    self.data[key] = df
        else:
            # Load from files as before
            files = {
                'schedules': 'cleaned_schedules after 2024.csv',
                'delays': 'cleaned_technical_delay.csv',
                'actions': 'cleaned_technical_delay_action.csv',
                'airlines': 'cleaned_airline.csv',
                'aircraft_types': 'cleaned_aircraft_type.csv',
                'registrations': 'cleaned_aircraft_registration.csv',
                'companies': 'cleaned_company.csv',
                'clusters': 'cleaned_aircraft_type_cluster.csv'
            }

            for key, filename in files.items():
                file_path = os.path.join(self.data_path, filename)
                try:
                    if key == 'schedules':
                        df = pd.read_csv(file_path, quotechar='"', escapechar='\\')
                    else:
                        df = pd.read_csv(file_path)

                    # Convert datetime columns
                    for col in df.columns:
                        if any(word in col.lower() for word in ['time', 'date', 'created_on', 'updated_on']):
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                    self.data[key] = df
                    print(f"✓ Loaded {key}: {df.shape[0]:,} rows, {df.shape[1]} columns")
                except Exception as e:
                    print(f"✗ Error loading {filename}: {e}")
                    self.data[key] = pd.DataFrame()

    def get_data_status(self):
        """Get status of loaded data for dashboard display"""
        status = {}
        for key, df in self.data.items():
            if df.empty:
                status[key] = "Not available"
            else:
                status[key] = f"{df.shape[0]:,} rows, {df.shape[1]} columns"
        return status

    def calculate_delay_hours(self):
        """Calculate actual delay hours from datetime columns"""
        if self.data['delays'].empty:
            return pd.DataFrame()

        delays = self.data['delays'].copy()

        # Calculate actual delay duration in hours with error handling
        delays['call_time_dt'] = pd.to_datetime(delays['call_time'], errors='coerce')
        delays['end_of_work_time_dt'] = pd.to_datetime(delays['end_of_work_time'], errors='coerce')
        delays['delay_duration_hours'] = (
            delays['end_of_work_time_dt'] - delays['call_time_dt']
        ).dt.total_seconds() / 3600

        # Filter out invalid delays
        delays = delays[
            (delays['delay_duration_hours'] > 0) &
            (delays['delay_duration_hours'] < 100)
        ].copy()

        # Extract aircraft registration from flight number
        delays['aircraft_reg'] = delays['departure_flight_number'].str.extract(r'([A-Z]{2,3}\d{3,4})')

        return delays

    def merge_with_aircraft_info(self, delays):
        """Merge delays with aircraft information"""
        if delays.empty or self.data['registrations'].empty:
            return delays

        merged = delays.merge(
            self.data['registrations'][['aircraft_registration', 'aircraft_id', 'aircraft_engine_type_code']],
            left_on='aircraft_reg',
            right_on='aircraft_registration',
            how='left'
        )

        if not self.data['aircraft_types'].empty:
            merged = merged.merge(
                self.data['aircraft_types'][['aircraft_id', 'aircraft_type', 'engine_count']],
                on='aircraft_id',
                how='left'
            )

        return merged

    def get_klm_airlines(self):
        """Get KLM-related airline IDs"""
        if 'airlines' in self.data and not self.data['airlines'].empty:
            klm_airlines = self.data['airlines'][
                self.data['airlines']['airline_name'].str.contains('KLM', case=False, na=False)
            ]
            return klm_airlines['airline_id'].tolist()
        return [132, 67, 193]  # Fallback based on our analysis

    def filter_klm_data(self, data, airline_id_col='airline_id'):
        """Filter dataframe to only include KLM data using company_id == 1"""
        if data.empty:
            return data

        # Filter by company_id == 1 (KLM)
        if 'company_id' in data.columns:
            return data[data['company_id'] == 1].copy()

        # For schedules data with airline_id, we need to join with schedules to get company_id
        if airline_id_col in data.columns and not self.data['schedules'].empty:
            # Get KLM schedules (company_id == 1)
            klm_schedules = self.data['schedules'][self.data['schedules']['company_id'] == 1]
            klm_airline_ids = klm_schedules['airline_id'].unique()
            return data[data[airline_id_col].isin(klm_airline_ids)].copy()

        # For delays data, filter via flight number matching with KLM schedules
        if 'departure_flight_number' in data.columns and not self.data['schedules'].empty:
            klm_schedules = self.data['schedules'][self.data['schedules']['company_id'] == 1]
            # Extract flight numbers from KLM schedules
            klm_flight_numbers = []
            for _, schedule in klm_schedules.iterrows():
                # Add both outbound and inbound flight numbers if they exist
                if pd.notna(schedule.get('outbound_flight_number')) and schedule['outbound_flight_number'] != 'Unknown':
                    klm_flight_numbers.append(str(schedule['outbound_flight_number']))
                if pd.notna(schedule.get('correct_flight_number')) and schedule['correct_flight_number'] != 'N/A':
                    klm_flight_numbers.append(str(schedule['correct_flight_number']))

            if klm_flight_numbers:
                filtered_data = data[data['departure_flight_number'].astype(str).isin(klm_flight_numbers)].copy()
                return filtered_data

        # For actions data - the technical_delay_action.csv is already KLM filtered, so return as-is
        if 'technical_delay_id' in data.columns:
            return data.copy()

        # If no direct filtering possible, return empty dataframe
        return pd.DataFrame()

def create_enhanced_box_plot_with_stats(data, y_col, title, color_col=None, height=400):
    """Create enhanced box plot with statistical cards"""

    # Create box plot
    if color_col and color_col in data.columns:
        fig = go.Figure()

        for category in data[color_col].unique():
            category_data = data[data[color_col] == category][y_col].dropna()
            if len(category_data) >= 3:  # Only show categories with enough data
                fig.add_trace(go.Box(
                    y=category_data,
                    name=str(category),
                    boxpoints='outliers',
                    marker_color=px.colors.qualitative.Set1[list(data[color_col].unique()).index(category) % len(px.colors.qualitative.Set1)]
                ))
    else:
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=data[y_col].dropna(),
            boxpoints='outliers',
            marker_color='#0066CC'
        ))

    fig.update_layout(
        title=title,
        yaxis_title=y_col.replace('_', ' ').title(),
        height=height,
        showlegend=True if color_col else False
    )

    # Calculate statistics
    stats_data = data[y_col].dropna()
    if len(stats_data) > 0:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{stats_data.min():.2f}</div>
                <div class="stats-label">Minimum</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{stats_data.quantile(0.25):.2f}</div>
                <div class="stats-label">Q1 (25%)</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{stats_data.median():.2f}</div>
                <div class="stats-label">Median</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{stats_data.quantile(0.75):.2f}</div>
                <div class="stats-label">Q3 (75%)</div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{stats_data.max():.2f}</div>
                <div class="stats-label">Maximum</div>
            </div>
            """, unsafe_allow_html=True)

    return fig

def add_explanation(title, explanation, data_source="", columns_used="", calculation_method="", filters_applied=""):
    """Add detailed explanation box for visualizations with data source information"""

    # Build detailed explanation
    detailed_info = f"<strong>{title}</strong><br><br>"
    detailed_info += f"<strong>What this visualization shows:</strong><br>{explanation}<br><br>"

    if data_source:
        detailed_info += f"<strong>Data Source:</strong> {data_source}<br>"

    if columns_used:
        detailed_info += f"<strong>Columns Used:</strong> {columns_used}<br>"

    if calculation_method:
        detailed_info += f"<strong>Calculation Method:</strong> {calculation_method}<br>"

    if filters_applied:
        detailed_info += f"<strong>Filters Applied:</strong> {filters_applied}<br>"

    st.markdown(f"""
    <div class="explanation-box">
        {detailed_info}
    </div>
    """, unsafe_allow_html=True)

def create_executive_dashboard(processor):
    """Executive dashboard with KPIs and high level overview"""
    st.markdown('<h1 class="main-header">Executive Operations Dashboard</h1>', unsafe_allow_html=True)

    schedules = processor.data['schedules']
    delays = processor.calculate_delay_hours()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_flights = len(schedules)
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{total_flights:,}</div>
            <div class="metric-label">Total Flights</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        total_delays = len(delays)
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{total_delays:,}</div>
            <div class="metric-label">Technical Delays</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if not delays.empty:
            avg_delay = delays['delay_duration_hours'].mean()
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{avg_delay:.1f}h</div>
                <div class="metric-label">Average Delay</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">0h</div>
                <div class="metric-label">Average Delay</div>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        if not schedules.empty:
            unique_airlines = schedules['airline_id'].nunique()
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{unique_airlines}</div>
                <div class="metric-label">Active Airlines</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">0</div>
                <div class="metric-label">Active Airlines</div>
            </div>
            """, unsafe_allow_html=True)

    add_explanation("Executive Overview",
                   "High-level operational metrics showing total flight volume, delay incidents, and airline activity across the entire dataset.",
                   "cleaned_schedules after 2024.csv, cleaned_technical_delay.csv",
                   "scheduled_departure_time, airline_id, call_time, end_of_work_time",
                   "Total flights = COUNT(all rows in schedules), Total delays = COUNT(all rows in delays), Average delay = AVERAGE(end_of_work_time - call_time in hours), Active airlines = COUNT(DISTINCT airline_id)",
                   "None (all data shown)")

    # Monthly trend
    if not schedules.empty:
        st.markdown('<h2 class="section-header">Monthly Operational Trends</h2>', unsafe_allow_html=True)

        schedules['month'] = schedules['scheduled_departure_time'].dt.month
        monthly_flights = schedules.groupby('month').size().reset_index(name='flight_count')

        month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                      7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
        monthly_flights['month_name'] = monthly_flights['month'].map(month_names)

        fig = px.line(
            monthly_flights,
            x='month_name',
            y='flight_count',
            title="Flight Volume per Month",
            labels={'month_name': 'Month', 'flight_count': 'Number of Flights'},
            markers=True
        )
        fig.update_traces(line_color='#0066CC', line_width=3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        date_range = f"{schedules['scheduled_departure_time'].min().strftime('%Y-%m')} to {schedules['scheduled_departure_time'].max().strftime('%Y-%m')}"
        add_explanation("Monthly Flight Trends",
                       "Shows seasonal patterns in flight volume throughout the year, helping identify peak travel periods and operational planning needs.",
                       "cleaned_schedules after 2024.csv",
                       "scheduled_departure_time (extracted month)",
                       "Flights per month = COUNT(DISTINCT flight_schedule_id) GROUP BY month",
                       f"Date range: {date_range}")

def create_technical_performance_dashboard(processor):
    """Consolidated technical performance dashboard with defect and action analysis"""
    st.markdown('<h1 class="main-header">Technical Performance & Defect Analysis</h1>', unsafe_allow_html=True)

    delays = processor.calculate_delay_hours()
    # Filter delays for KLM
    delays = processor.filter_klm_data(delays)
    delays_merged = processor.merge_with_aircraft_info(delays)
    actions = processor.data['actions']  # Already KLM filtered

    if delays.empty:
        st.warning("No delay data available for analysis")
        return

    # Action analysis if available (moved to top)
    if not actions.empty:
        st.markdown('<h2 class="section-header"> Action Analysis</h2>', unsafe_allow_html=True)

        # Actions per delay analysis
        actions_per_delay = actions.groupby('technical_delay_id').size().reset_index(name='action_count')

        # Calculate metrics
        total_delays_with_actions = len(actions_per_delay)
        rectification_count = actions_per_delay[actions_per_delay['action_count'] == 1].shape[0]
        troubleshooting_count = actions_per_delay[actions_per_delay['action_count'] > 1].shape[0]
        avg_actions = actions_per_delay['action_count'].mean()

        # Display metrics in four columns for better spacing
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{rectification_count:,}</div>
                <div class="metric-label">Rectification</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem;">
                    Single action delays<br>
                    ({(rectification_count/total_delays_with_actions)*100:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{troubleshooting_count:,}</div>
                <div class="metric-label">Troubleshooting</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem;">
                    Multiple action delays<br>
                    ({(troubleshooting_count/total_delays_with_actions)*100:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{avg_actions:.1f}</div>
                <div class="metric-label">Avg Actions per Delay</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{total_delays_with_actions:,}</div>
                <div class="metric-label">Total Delays with Actions</div>
            </div>
            """, unsafe_allow_html=True)

        add_explanation("Action Types Analysis",
                       "Quantitative analysis of maintenance actions per technical delay (technical_delay_id). Classifies complexity of repair procedures - rectification (single action) versus troubleshooting (multiple actions). Essential for resource planning, technical staffing, and maintenance process effectiveness measurement.",
                       "cleaned_technical_delay.csv + cleaned_technical_delay_action.csv (joined on technical_delay_id)",
                       "technical_delay_id, action_id, action_count (calculated)",
                       "Action analysis = COUNT(action_id) GROUP BY technical_delay_id, calculation percentages: rectification = WHERE action_count = 1, troubleshooting = WHERE action_count > 1, average = MEAN(action_count)",
                       "Technical delays with registered actions, classification in rectification (1 action) and troubleshooting (>1 action)")

    # Delay patterns analysis
    st.markdown('<h2 class="section-header">Delay Distribution Analysis</h2>', unsafe_allow_html=True)

    # Delay Duration Categories - Operational Impact Assessment

    delay_categories = pd.cut(
        delays['delay_duration_hours'],
        bins=[0, 2, 6, 12, 24, float('inf')],
        labels=['< 2h', '2-6h', '6-12h', '12-24h', '> 24h']
    )
    category_counts = delay_categories.value_counts()

    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Delay Categories by Duration"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    add_explanation("Delay Duration Categories",
                   "Distribution of technical delays by operational impact categories, enabling resource planning and operational response strategy development.",
                   "cleaned_technical_delay.csv (KLM filtered)",
                   "call_time, end_of_work_time",
                   "delay_duration_hours = (end_of_work_time - call_time) / 3600, then categorized into operational impact bins: <2h (minor), 2-6h (moderate), 6-12h (significant), 12-24h (severe), >24h (critical)",
                   "Valid KLM delays only (duration > 0 and < 100 hours, aircraft_reg successfully extracted)")

    # Defect analysis (consolidated from 3 duplicates)
    st.markdown('<h2 class="section-header">Defect Analysis</h2>', unsafe_allow_html=True)

    # Top defect categories by average delay
    defect_analysis = delays.groupby('ata_chapter_name')['delay_duration_hours'].agg(['mean', 'count']).reset_index()
    defect_analysis = defect_analysis[defect_analysis['count'] >= 3]
    defect_analysis = defect_analysis.sort_values('mean', ascending=False).head(10)

    fig = px.bar(
        defect_analysis,
        x='mean',
        y='ata_chapter_name',
        orientation='h',
        title="Top 10 Defect Categories by Average Delay",
        labels={'mean': 'Average Delay (Hours)', 'ata_chapter_name': 'Defect Category'},
        color='mean',
        color_continuous_scale='oranges'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    add_explanation("Defect Categories - Average Impact",
                   "Shows which defect types cause the longest delays, crucial for maintenance planning and spare parts inventory.",
                   "cleaned_technical_delay.csv",
                   "ata_chapter_name, delay_duration_hours",
                   "Average delay = MEAN(delay_duration_hours) GROUP BY ata_chapter_name, then sorted by highest average, limited to categories with 3+ incidents and top 10 results",
                   "Minimum 3 incidents per category, top 10 by average delay")

    # Most frequent defect categories
    defect_freq = delays['ata_chapter_name'].value_counts().head(10)
    defect_freq_df = pd.DataFrame({'defect_category': defect_freq.index, 'count': defect_freq.values})

    fig = px.bar(
        defect_freq_df,
        x='count',
        y='defect_category',
        orientation='h',
        title="Top 10 Most Frequent Defect Categories",
        labels={'count': 'Number of Delays', 'defect_category': 'Defect Category'},
        color='count',
        color_continuous_scale='purples'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    add_explanation("Defect Categories - Frequency",
                   "Shows most common defect types, important for training programs and prevention strategies.",
                   "cleaned_technical_delay.csv",
                   "ata_chapter_name",
                   "Frequency = COUNT(incidents) GROUP BY ata_chapter_name, then sorted by highest count, limited to top 10",
                   "Top 10 most frequent categories")

    # Most Common Defect Categories
    st.markdown('<h2 class="section-header">Most Common Defect Categories</h2>', unsafe_allow_html=True)

    # Most frequent delay codes
    defect_codes = delays['delay_code_description1'].value_counts().head(15)
    defect_df = pd.DataFrame({'defect_code': defect_codes.index, 'count': defect_codes.values})

    fig = px.bar(
        defect_df,
        x='count',
        y='defect_code',
        orientation='h',
        title="Top 15 Defect Categories",
        labels={'count': 'Number of Delays', 'defect_code': 'Defect Category'},
        color='count',
        color_continuous_scale='oranges'
    )
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    add_explanation("Most Frequent Defect Categories",
                   "Analysis of most common defect types based on delay_code_description1. Frequency analysis identifies systematic technical problems that repeatedly occur in fleet, crucial for preventive maintenance and training programs.",
                   "cleaned_technical_delay.csv",
                   "delay_code_description1, technical_delay_id, delay_duration_hours",
                   "Frequency analysis = COUNT(delay_code_description1) GROUP BY defect category, sorted by highest frequency, TOP 15 results",
                   "Technical delays with valid delay_code_description1, limited to top 15 most common defects")

    # Average delay by defect category
    defect_avg = delays.groupby('delay_code_description1')['delay_duration_hours'].agg(['mean', 'count']).reset_index()
    defect_avg = defect_avg[defect_avg['count'] >= 3]  # Filter categories with enough data
    defect_avg = defect_avg.sort_values('mean', ascending=False).head(10)

    fig = px.bar(
        defect_avg,
        x='mean',
        y='delay_code_description1',
        orientation='h',
        title="Average Delay by Defect Category",
        labels={'mean': 'Average Delay (Hours)', 'delay_code_description1': 'Defect Category'},
        color='mean',
        color_continuous_scale='reds'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    add_explanation("Average Delay by Defect Category",
                   "Weighted average analysis per defect type based on delay_code_description1 and actual delay_duration_hours. Identifies defect categories with highest operational impact, essential for maintenance resource prioritization and critical path analysis.",
                   "cleaned_technical_delay.csv",
                   "delay_code_description1, delay_duration_hours, technical_delay_id",
                   "Average calculation = MEAN(delay_duration_hours) GROUP BY delay_code_description1, filter on categories with ≥3 incidents for statistical reliability, sort by highest average, TOP 10 results",
                   "Technical delays with valid duration (>0 hours), minimum 3 incidents per category for statistical validity, top 10 by average impact")

    # Aircraft performance (consolidated)
    if not delays_merged.empty:
        st.markdown('<h2 class="section-header">Aircraft Performance</h2>', unsafe_allow_html=True)

        if 'aircraft_reg' in delays_merged.columns:
            aircraft_delays = delays_merged.groupby('aircraft_reg')['delay_duration_hours'].agg(['mean', 'count']).reset_index()
            aircraft_delays = aircraft_delays[aircraft_delays['count'] >= 3]
            aircraft_delays = aircraft_delays.sort_values('mean', ascending=False).head(10)

            if not aircraft_delays.empty:
                fig = px.bar(
                    aircraft_delays,
                    x='mean',
                    y='aircraft_reg',
                    orientation='h',
                    title="Top 10 Aircraft by Average Delay",
                    labels={'mean': 'Average Delay (Hours)', 'aircraft_reg': 'Aircraft Registration'},
                    color='mean',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)

                add_explanation("Aircraft Performance by Registration",
                               "Identifies individual aircraft with the most technical problems, helping maintenance scheduling and fleet planning decisions.",
                               "cleaned_technical_delay.csv + cleaned_aircraft_registration.csv + cleaned_aircraft_type.csv (merged)",
                               "departure_flight_number (extracted aircraft_reg), delay_duration_hours, aircraft_id, aircraft_type",
                               "Average delay = MEAN(delay_duration_hours) GROUP BY aircraft_reg, then sorted by highest average, limited to aircraft with 3+ incidents and top 10 results",
                               "Minimum 3 incidents per aircraft, top 10 by average delay")

def create_operations_fleet_dashboard(processor):
    """Consolidated operations and fleet dashboard - KLM focused"""
    st.markdown('<h1 class="main-header">KLM Operations & Fleet Analytics</h1>', unsafe_allow_html=True)

    # KLM banner context
    st.markdown("""
    <div style="background: linear-gradient(90deg, #0066CC, #004499); color: white; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
        <strong>KLM Royal Dutch Airlines</strong> - Operational analysis of KLM fleet and flights
    </div>
    """, unsafe_allow_html=True)

    schedules = processor.filter_klm_data(processor.data['schedules'])
    registrations = processor.filter_klm_data(processor.data['registrations'])
    aircraft_types = processor.data['aircraft_types']

    # Schedule patterns (consolidated from 2 duplicates)
    if not schedules.empty:
        st.markdown('<h2 class="section-header">Schedule Patterns</h2>', unsafe_allow_html=True)

        # Day of week analysis
        def parse_day_string(day_str):
            if pd.isna(day_str) or day_str == '':
                return []

            day_str = str(day_str).strip().strip('{}').replace(' ', '')

            day_mapping = {
                'MON': 'Monday', 'TUE': 'Tuesday', 'WED': 'Wednesday', 'THU': 'Thursday',
                'FRI': 'Friday', 'SAT': 'Saturday', 'SUN': 'Sunday',
                'Mon': 'Monday', 'Tue': 'Tuesday', 'Wed': 'Wednesday', 'Thu': 'Thursday',
                'Fri': 'Friday', 'Sat': 'Saturday', 'Sun': 'Sunday',
                'MONDAY': 'Monday', 'TUESDAY': 'Tuesday', 'WEDNESDAY': 'Wednesday',
                'THURSDAY': 'Thursday', 'FRIDAY': 'Friday', 'SATURDAY': 'Saturday', 'SUNDAY': 'Sunday'
            }

            days_in_str = [d.strip() for d in day_str.split(',') if d.strip()]
            parsed_days = []

            for day in days_in_str:
                upper_day = day.upper()
                if upper_day in day_mapping:
                    parsed_days.append(day_mapping[upper_day])
                elif day in day_mapping.values():
                    parsed_days.append(day)
                else:
                    for code, full_name in day_mapping.items():
                        if code in upper_day or full_name.upper() in upper_day:
                            parsed_days.append(full_name)
                            break

            return parsed_days

        all_days = []
        for day_str in schedules['day_of_operating'].dropna():
            parsed_days = parse_day_string(day_str)
            all_days.extend(parsed_days)

        day_counts = Counter(all_days)
        ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts_ordered = [(day, day_counts.get(day, 0)) for day in ordered_days]

        day_df = pd.DataFrame(day_counts_ordered, columns=['Day', 'Flight Count'])

        fig = px.bar(
            day_df,
            x='Day',
            y='Flight Count',
            title="Flights per Day",
            labels={'Day': 'Day', 'Flight Count': 'Number of Flights'},
            color='Flight Count',
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

        add_explanation("Daily Flight Patterns",
                       "Shows flight distribution throughout the week, crucial for crew planning and resource allocation.",
                       "cleaned_schedules after 2024.csv (KLM filtered)",
                       "day_of_operating",
                       "Parse day strings from various formats, count flights per day, display in weekday order",
                       "KLM flights only")

        # Hourly distribution
        schedules['departure_hour'] = schedules['scheduled_departure_time'].dt.hour
        hourly_flights = schedules['departure_hour'].value_counts().sort_index()

        fig = px.line(
            x=hourly_flights.index,
            y=hourly_flights.values,
            title="Flight Distribution per Hour",
            labels={'x': 'Hour', 'y': 'Number of Flights'},
            markers=True
        )
        fig.update_traces(line_color='#0066CC', line_width=3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        add_explanation("Hourly Flight Patterns",
                       "Shows peak hours and quiet periods, important for gate planning and ground handling operations.",
                       "cleaned_schedules after 2024.csv (KLM filtered)",
                       "scheduled_departure_time (extracted hour)",
                       "Extract hour from scheduled_departure_time, count flights per hour, sort by hour",
                       "KLM flights only")

    # Interactive World Map with All Inbound Stations as Pins
    if not schedules.empty and 'inbound_station_code' in schedules.columns:
        st.markdown('<h2 class="section-header">Global Inbound Stations Map</h2>', unsafe_allow_html=True)

        # Load stations data
        try:
            stations_data = pd.read_csv(os.path.join(processor.data_path, 'cleaned_All Stations WorldWide (sharepointlist)-1.csv'))

            # Count flights per inbound station
            station_counts = schedules['inbound_station_code'].value_counts().reset_index()
            station_counts.columns = ['Station code', 'flight_count']

            # Merge with station information
            station_counts = station_counts.merge(
                stations_data,
                on='Station code',
                how='left'
            ).dropna(subset=['Station name'])

            # Try a simple world map approach
            try:

                # Create comprehensive airport coordinates database with hundreds of airports
                known_locations = {
                    # Major European hubs
                    'CDG': {'lat': 49.0097, 'lon': 2.5479, 'name': 'Paris Charles De Gaulle'},
                    'LHR': {'lat': 51.4700, 'lon': -0.4543, 'name': 'London Heathrow'},
                    'AMS': {'lat': 52.3105, 'lon': 4.7683, 'name': 'Amsterdam Schiphol'},
                    'FRA': {'lat': 50.0379, 'lon': 8.5622, 'name': 'Frankfurt Main'},
                    'MUC': {'lat': 48.3538, 'lon': 11.7861, 'name': 'Munich'},
                    'ZRH': {'lat': 47.4647, 'lon': 8.5492, 'name': 'Zurich'},
                    'VIE': {'lat': 48.1103, 'lon': 16.5697, 'name': 'Vienna'},
                    'BRU': {'lat': 50.9014, 'lon': 4.4844, 'name': 'Brussels'},
                    'MAD': {'lat': 40.4983, 'lon': -3.5676, 'name': 'Madrid Barajas'},
                    'BCN': {'lat': 41.2971, 'lon': 2.0785, 'name': 'Barcelona'},
                    'FCO': {'lat': 41.8003, 'lon': 12.2389, 'name': 'Rome Fiumicino'},
                    'CPH': {'lat': 55.6181, 'lon': 12.6561, 'name': 'Copenhagen'},
                    'ARN': {'lat': 59.6497, 'lon': 17.9217, 'name': 'Stockholm Arlanda'},
                    'OSL': {'lat': 60.1939, 'lon': 11.1004, 'name': 'Oslo'},
                    'HEL': {'lat': 60.3172, 'lon': 24.9633, 'name': 'Helsinki'},
                    'WAW': {'lat': 52.1657, 'lon': 20.9671, 'name': 'Warsaw'},
                    'PRG': {'lat': 50.1008, 'lon': 14.2632, 'name': 'Prague'},
                    'BUD': {'lat': 47.4355, 'lon': 19.2547, 'name': 'Budapest'},
                    'IST': {'lat': 41.0151, 'lon': 28.9795, 'name': 'Istanbul'},
                    'ATH': {'lat': 37.9364, 'lon': 23.9445, 'name': 'Athens'},
                    'LIS': {'lat': 38.7813, 'lon': -9.1359, 'name': 'Lisbon'},
                    'OPO': {'lat': 41.2481, 'lon': -8.6814, 'name': 'Porto'},
                    'DUB': {'lat': 53.4213, 'lon': -6.2700, 'name': 'Dublin'},
                    'MAN': {'lat': 53.3537, 'lon': -2.2749, 'name': 'Manchester'},
                    'GVA': {'lat': 46.2381, 'lon': 6.1089, 'name': 'Geneva'},
                    'NCE': {'lat': 43.6584, 'lon': 7.2158, 'name': 'Nice'},
                    'MXP': {'lat': 45.6306, 'lon': 8.7281, 'name': 'Milan Malpensa'},
                    'LGW': {'lat': 51.1537, 'lon': -0.1821, 'name': 'London Gatwick'},
                    'STN': {'lat': 51.8850, 'lon': 0.2354, 'name': 'London Stansted'},
                    'EDI': {'lat': 55.9463, 'lon': -3.3717, 'name': 'Edinburgh'},
                    'GLA': {'lat': 55.8719, 'lon': -4.4331, 'name': 'Glasgow'},
                    'BHX': {'lat': 52.4539, 'lon': -1.7481, 'name': 'Birmingham'},
                    # Major North American airports
                    'JFK': {'lat': 40.6413, 'lon': -73.7781, 'name': 'New York JFK'},
                    'LAX': {'lat': 33.9425, 'lon': -118.4081, 'name': 'Los Angeles'},
                    'ORD': {'lat': 41.9742, 'lon': -87.9073, 'name': 'Chicago O\'Hare'},
                    'ATL': {'lat': 33.6407, 'lon': -84.4277, 'name': 'Atlanta'},
                    'DFW': {'lat': 32.8998, 'lon': -97.0403, 'name': 'Dallas Fort Worth'},
                    'DEN': {'lat': 39.8617, 'lon': -104.6731, 'name': 'Denver'},
                    'SFO': {'lat': 37.6213, 'lon': -122.3790, 'name': 'San Francisco'},
                    'SEA': {'lat': 47.4502, 'lon': -122.3088, 'name': 'Seattle'},
                    'LAS': {'lat': 36.0840, 'lon': -115.1537, 'name': 'Las Vegas'},
                    'MIA': {'lat': 25.7959, 'lon': -80.2870, 'name': 'Miami'},
                    'BOS': {'lat': 42.3656, 'lon': -71.0096, 'name': 'Boston Logan'},
                    'IAD': {'lat': 38.9531, 'lon': -77.4565, 'name': 'Washington Dulles'},
                    'DCA': {'lat': 38.8512, 'lon': -77.0402, 'name': 'Washington Reagan'},
                    'PHL': {'lat': 39.8719, 'lon': -75.2411, 'name': 'Philadelphia'},
                    'PHX': {'lat': 33.4484, 'lon': -112.0740, 'name': 'Phoenix'},
                    'IAH': {'lat': 29.9902, 'lon': -95.3368, 'name': 'Houston Bush'},
                    'MSP': {'lat': 44.8848, 'lon': -93.2223, 'name': 'Minneapolis'},
                    'DTW': {'lat': 42.2124, 'lon': -83.3534, 'name': 'Detroit'},
                    'CLT': {'lat': 35.2144, 'lon': -80.9473, 'name': 'Charlotte'},
                    'BWI': {'lat': 39.1754, 'lon': -76.6683, 'name': 'Baltimore'},
                    'SLC': {'lat': 40.7899, 'lon': -111.9791, 'name': 'Salt Lake City'},
                    'SAN': {'lat': 32.7336, 'lon': -117.1897, 'name': 'San Diego'},
                    'TPA': {'lat': 27.9755, 'lon': -82.5332, 'name': 'Tampa'},
                    'MCO': {'lat': 28.4312, 'lon': -81.3081, 'name': 'Orlando'},
                    'FLL': {'lat': 26.0742, 'lon': -80.1506, 'name': 'Fort Lauderdale'},
                    'LGA': {'lat': 40.7769, 'lon': -73.8740, 'name': 'New York LaGuardia'},
                    'EWR': {'lat': 40.6925, 'lon': -74.1687, 'name': 'Newark'},
                    # Canadian airports
                    'YYZ': {'lat': 43.6777, 'lon': -79.6248, 'name': 'Toronto Pearson'},
                    'YVR': {'lat': 49.1967, 'lon': -123.1815, 'name': 'Vancouver'},
                    'YUL': {'lat': 45.4706, 'lon': -73.7408, 'name': 'Montreal'},
                    'YYC': {'lat': 51.1304, 'lon': -114.0199, 'name': 'Calgary'},
                    'YEG': {'lat': 53.3097, 'lon': -113.5801, 'name': 'Edmonton'},
                    'YOW': {'lat': 45.3225, 'lon': -75.6692, 'name': 'Ottawa'},
                    'YWG': {'lat': 49.9100, 'lon': -97.2398, 'name': 'Winnipeg'},
                    'YHZ': {'lat': 44.8808, 'lon': -63.5087, 'name': 'Halifax'},
                    # Asian airports
                    'NRT': {'lat': 35.7653, 'lon': 140.3864, 'name': 'Tokyo Narita'},
                    'HND': {'lat': 35.5533, 'lon': 139.7811, 'name': 'Tokyo Haneda'},
                    'ICN': {'lat': 37.4631, 'lon': 126.4406, 'name': 'Seoul Incheon'},
                    'GMP': {'lat': 37.5583, 'lon': 126.7906, 'name': 'Seoul Gimpo'},
                    'PVG': {'lat': 31.1434, 'lon': 121.8055, 'name': 'Shanghai Pudong'},
                    'SHA': {'lat': 31.1979, 'lon': 121.3364, 'name': 'Shanghai Hongqiao'},
                    'PEK': {'lat': 40.0801, 'lon': 116.5846, 'name': 'Beijing Capital'},
                    'PKX': {'lat': 39.8094, 'lon': 116.4105, 'name': 'Beijing Daxing'},
                    'CAN': {'lat': 23.3924, 'lon': 113.2988, 'name': 'Guangzhou'},
                    'SZX': {'lat': 22.6393, 'lon': 113.8109, 'name': 'Shenzhen'},
                    'HKG': {'lat': 22.3080, 'lon': 113.9185, 'name': 'Hong Kong'},
                    'SIN': {'lat': 1.3644, 'lon': 103.9915, 'name': 'Singapore Changi'},
                    'BKK': {'lat': 13.6900, 'lon': 100.7501, 'name': 'Bangkok Suvarnabhumi'},
                    'DMK': {'lat': 13.9126, 'lon': 100.6065, 'name': 'Bangkok Don Mueang'},
                    'KUL': {'lat': 2.7456, 'lon': 101.7072, 'name': 'Kuala Lumpur'},
                    'CGK': {'lat': -6.1256, 'lon': 106.6558, 'name': 'Jakarta Soekarno-Hatta'},
                    'MNL': {'lat': 14.5086, 'lon': 121.0194, 'name': 'Manila'},
                    'TPE': {'lat': 25.0777, 'lon': 121.2328, 'name': 'Taipei Taoyuan'},
                    'DEL': {'lat': 28.5665, 'lon': 77.1031, 'name': 'New Delhi'},
                    'BOM': {'lat': 19.0896, 'lon': 72.8656, 'name': 'Mumbai'},
                    'BLR': {'lat': 13.1986, 'lon': 77.7066, 'name': 'Bangalore'},
                    'CCU': {'lat': 22.6547, 'lon': 88.4461, 'name': 'Kolkata'},
                    'MAA': {'lat': 12.9941, 'lon': 80.1809, 'name': 'Chennai'},
                    # Middle Eastern airports
                    'DXB': {'lat': 25.2532, 'lon': 55.3657, 'name': 'Dubai'},
                    'DOH': {'lat': 25.2731, 'lon': 51.6088, 'name': 'Doha'},
                    'AUH': {'lat': 24.4330, 'lon': 54.6511, 'name': 'Abu Dhabi'},
                    'JED': {'lat': 21.6700, 'lon': 39.1465, 'name': 'Jeddah'},
                    'RUH': {'lat': 24.9576, 'lon': 46.6988, 'name': 'Riyadh'},
                    'KWI': {'lat': 29.2267, 'lon': 47.9689, 'name': 'Kuwait'},
                    'MCT': {'lat': 23.5934, 'lon': 58.2844, 'name': 'Muscat'},
                    'BGW': {'lat': 33.2630, 'lon': 44.2420, 'name': 'Baghdad'},
                    'IKA': {'lat': 35.4169, 'lon': 51.1819, 'name': 'Tehran Imam Khomeini'},
                    'TLV': {'lat': 32.0094, 'lon': 34.8835, 'name': 'Tel Aviv'},
                    'AMM': {'lat': 31.7224, 'lon': 35.9876, 'name': 'Amman'},
                    # African airports
                    'CAI': {'lat': 30.1219, 'lon': 31.4056, 'name': 'Cairo'},
                    'JNB': {'lat': -26.1367, 'lon': 28.2411, 'name': 'Johannesburg OR Tambo'},
                    'CPT': {'lat': -33.9691, 'lon': 18.6017, 'name': 'Cape Town'},
                    'LOS': {'lat': 6.5774, 'lon': 3.3212, 'name': 'Lagos Murtala Muhammed'},
                    'NBO': {'lat': -1.3192, 'lon': 36.9278, 'name': 'Nairobi Jomo Kenyatta'},
                    'ADD': {'lat': 8.9806, 'lon': 38.7997, 'name': 'Addis Ababa'},
                    'DKR': {'lat': 14.7396, 'lon': -17.4906, 'name': 'Dakar'},
                    'CMN': {'lat': 33.3675, 'lon': -7.5897, 'name': 'Casablanca'},
                    'TUN': {'lat': 36.8489, 'lon': 10.2014, 'name': 'Tunis'},
                    'ALG': {'lat': 36.6911, 'lon': 3.2151, 'name': 'Algiers'},
                    'RAB': {'lat': 34.0519, 'lon': -6.7958, 'name': 'Rabat'},
                    'DAR': {'lat': -6.8684, 'lon': 39.2037, 'name': 'Dar es Salaam'},
                    'JRO': {'lat': -2.8333, 'lon': 37.0667, 'name': 'Kilimanjaro'},
                    'EBB': {'lat': 0.0424, 'lon': 32.4435, 'name': 'Entebbe'},
                    'KGL': {'lat': -1.9686, 'lon': 30.1343, 'name': 'Kigali'},
                    'FIH': {'lat': -4.3856, 'lon': 15.2543, 'name': 'Kinshasa'},
                    # South American airports
                    'GRU': {'lat': -23.4356, 'lon': -46.4731, 'name': 'São Paulo Guarulhos'},
                    'GIG': {'lat': -22.8089, 'lon': -43.2436, 'name': 'Rio de Janeiro Galeão'},
                    'EZE': {'lat': -34.8222, 'lon': -58.5358, 'name': 'Buenos Aires Ezeiza'},
                    'SCL': {'lat': -33.3930, 'lon': -70.7858, 'name': 'Santiago'},
                    'LIM': {'lat': -12.0219, 'lon': -77.1146, 'name': 'Lima'},
                    'BOG': {'lat': 4.7016, 'lon': -74.1469, 'name': 'Bogotá'},
                    'CCS': {'lat': 10.6019, 'lon': -66.9911, 'name': 'Caracas'},
                    'UIO': {'lat': -0.1292, 'lon': -78.3578, 'name': 'Quito'},
                    'GYE': {'lat': -2.1575, 'lon': -79.8836, 'name': 'Guayaquil'},
                    'AEP': {'lat': -34.5592, 'lon': -58.4156, 'name': 'Buenos Aires Aeroparque'},
                    'POA': {'lat': -29.9944, 'lon': -51.1714, 'name': 'Porto Alegre'},
                    'CNF': {'lat': -19.6244, 'lon': -43.9673, 'name': 'Belo Horizonte'},
                    'FOR': {'lat': -3.7763, 'lon': -38.5333, 'name': 'Fortaleza'},
                    'REC': {'lat': -8.1259, 'lon': -34.9236, 'name': 'Recife'},
                    'SSA': {'lat': -12.9111, 'lon': -38.3314, 'name': 'Salvador'},
                    # Australian and Pacific airports
                    'SYD': {'lat': -33.9399, 'lon': 151.1753, 'name': 'Sydney'},
                    'MEL': {'lat': -37.6690, 'lon': 144.8410, 'name': 'Melbourne'},
                    'BNE': {'lat': -27.3944, 'lon': 153.1173, 'name': 'Brisbane'},
                    'PER': {'lat': -31.9403, 'lon': 115.9667, 'name': 'Perth'},
                    'ADL': {'lat': -34.9459, 'lon': 138.5306, 'name': 'Adelaide'},
                    'OOL': {'lat': -28.1644, 'lon': 153.5029, 'name': 'Gold Coast'},
                    'CNS': {'lat': -16.8858, 'lon': 145.7553, 'name': 'Cairns'},
                    'DRW': {'lat': -12.4139, 'lon': 130.8765, 'name': 'Darwin'},
                    'HBA': {'lat': -42.8364, 'lon': 147.5084, 'name': 'Hobart'},
                    'AKL': {'lat': -37.0082, 'lon': 174.7850, 'name': 'Auckland'},
                    'CHC': {'lat': -43.4893, 'lon': 172.5344, 'name': 'Christchurch'},
                    'WLG': {'lat': -41.3272, 'lon': 174.8050, 'name': 'Wellington'},
                    'NAN': {'lat': -17.7553, 'lon': 177.4239, 'name': 'Nadi'},
                    'RAR': {'lat': -21.2078, 'lon': -159.8102, 'name': 'Rarotonga'},
                    'PPT': {'lat': -17.5535, 'lon': -149.6060, 'name': 'Tahiti'},
                    # Caribbean airports
                    'SJU': {'lat': 18.4383, 'lon': -66.0018, 'name': 'San Juan'},
                    'SDQ': {'lat': 18.4292, 'lon': -69.6727, 'name': 'Santo Domingo'},
                    'HAV': {'lat': 22.9892, 'lon': -82.4027, 'name': 'Havana'},
                    'KIN': {'lat': 17.9432, 'lon': -76.7831, 'name': 'Kingston'},
                    'MBJ': {'lat': 18.4966, 'lon': -77.9129, 'name': 'Montego Bay'},
                    'POS': {'lat': 10.6019, 'lon': -61.3502, 'name': 'Port of Spain'},
                    'BGI': {'lat': 13.0733, 'lon': -59.4934, 'name': 'Barbados'},
                    'ANU': {'lat': 17.1360, 'lon': -61.7931, 'name': 'Antigua'},
                    'GND': {'lat': 12.0078, 'lon': -61.7883, 'name': 'Grenada'},
                    'SLU': {'lat': 13.7432, 'lon': -60.9538, 'name': 'St Lucia'},
                    # Central American airports
                    'MEX': {'lat': 19.4363, 'lon': -99.0721, 'name': 'Mexico City'},
                    'GDL': {'lat': 20.5167, 'lon': -103.3117, 'name': 'Guadalajara'},
                    'MTY': {'lat': 25.7788, 'lon': -100.1079, 'name': 'Monterrey'},
                    'CUN': {'lat': 21.0411, 'lon': -86.8756, 'name': 'Cancun'},
                    'SJO': {'lat': 9.9976, 'lon': -84.2041, 'name': 'San José'},
                    'PTY': {'lat': 8.9836, 'lon': -79.5253, 'name': 'Panama City'},
                    'SAP': {'lat': 15.4470, 'lon': -88.0346, 'name': 'San Pedro Sula'},
                    'TGU': {'lat': 14.0565, 'lon': -87.2241, 'name': 'Tegucigalpa'},
                    'GUA': {'lat': 14.5856, 'lon': -90.5317, 'name': 'Guatemala City'},
                    'SAL': {'lat': 13.6988, 'lon': -89.6096, 'name': 'San Salvador'},
                    'MGA': {'lat': 12.1444, 'lon': -86.1851, 'name': 'Managua'}
                }

                # Filter stations that we have coordinates for
                map_stations = station_counts[station_counts['Station code'].isin(known_locations.keys())].copy()

                if len(map_stations) > 0:
                    # Add coordinates
                    map_stations['lat'] = map_stations['Station code'].map(lambda x: known_locations.get(x, {}).get('lat', 0))
                    map_stations['lon'] = map_stations['Station code'].map(lambda x: known_locations.get(x, {}).get('lon', 0))
                    map_stations['airport_name'] = map_stations['Station code'].map(lambda x: known_locations.get(x, {}).get('name', x))

                    
                    fig_map = px.scatter_mapbox(
                        map_stations,
                        lat='lat',
                        lon='lon',
                        hover_name='City name',
                        hover_data={
                            'Station name': True,
                            'Station code': True,
                            'flight_count': True,
                            'Country code': True
                        },
                        size='flight_count',
                        size_max=20,
                        color='flight_count',
                        color_continuous_scale='Viridis',
                        zoom=1,
                        title="🌍 Inbound Cities Map - Hover for details",
                        labels={'flight_count': 'Number of Flights'}
                    )

                    fig_map.update_layout(
                        mapbox_style="carto-positron",
                        height=500,
                        showlegend=True,
                        coloraxis_colorbar=dict(
                            title="Flight Count",
                            xanchor="left",
                            x=0.01
                        ),
                        margin=dict(l=0, r=0, t=50, b=0)
                    )

                    # Customize markers for better visibility
                    fig_map.update_traces(
                        marker=dict(
                            opacity=0.9
                        ),
                        selector=dict(mode='markers')
                    )

                    st.plotly_chart(fig_map, key="stations_world_map", use_container_width=True)
                else:
                    st.write("No known station locations found for mapping")

            except Exception as map_error:
                st.write(f"Map creation failed: {map_error}")

            # Show statistics below map
            col1, col2, col3 = st.columns(3)

            with col1:
                total_stations = len(station_counts)
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{total_stations:,}</div>
                    <div class="metric-label">Total Inbound Stations</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                total_flights = station_counts['flight_count'].sum()
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{total_flights:,}</div>
                    <div class="metric-label">Total Flights</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                avg_flights_per_station = station_counts['flight_count'].mean()
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{avg_flights_per_station:.1f}</div>
                    <div class="metric-label">Avg Flights per Station</div>
                </div>
                """, unsafe_allow_html=True)

            
            add_explanation("Global Inbound Stations Map",
                           "Interactive world map showing ALL airports from inbound_station_code as clickable pins. Each pin represents a unique destination airport. Click on any pin to see station details and total flight count to that destination.",
                           "cleaned_schedules after 2024.csv ",
                           "inbound_station_code (counted), Station code, Station name, City name, Country code",
                           "Count flights per station = COUNT(inbound_station_code) GROUP BY station, merge with stations database (hardcoded) for location info, display as uniform pins on world map",
                           "All stations in inbound_station_code, uniform pin size, searchable table with pagination")

        except Exception as e:
            st.error(f"Error loading stations data: {e}")
            st.info("Showing flight destination counts without map:")
            station_counts = schedules['inbound_station_code'].value_counts().head(50)
            st.dataframe(station_counts.reset_index().rename(columns={'index': 'Station Code', 'inbound_station_code': 'Flight Count'}))

    # Fleet analysis
    if not registrations.empty:
        st.markdown('<h2 class="section-header">Fleet Composition</h2>', unsafe_allow_html=True)

        # Engine types
        engine_types = registrations['aircraft_engine_type_code'].value_counts().head(10)

        fig = px.bar(
            x=engine_types.values,
            y=engine_types.index,
            orientation='h',
            title="Top 10 Engine Types",
            labels={'x': 'Number of Aircraft', 'y': 'Engine Type'},
            color=engine_types.values,
            color_continuous_scale='turbo'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

        add_explanation("Engine Type Distribution",
                       "Shows the distribution of engine types in the fleet. Important for maintenance planning and fuel efficiency analysis.",
                       "cleaned_aircraft_registration.csv",
                       "aircraft_engine_type_code",
                       "COUNT(aircraft_engine_type_code), ORDER BY count DESC LIMIT 10",
                       "None")

        # Aircraft Type Distribution - Overgenomen van Fleet Analytics
        if not aircraft_types.empty:
            st.markdown('<h2 class="section-header">Aircraft Type Distribution</h2>', unsafe_allow_html=True)

            fleet_with_types = registrations.merge(aircraft_types, on='aircraft_id', how='left')
            type_counts = fleet_with_types['aircraft_type'].value_counts()

            # Create pie chart for aircraft types distribution
            # For better visualization, let's group smaller types as "Others"
            top_types = type_counts.head(8)  # Show top 8 types individually
            others_count = type_counts.iloc[8:].sum() if len(type_counts) > 8 else 0

            # Prepare data for pie chart
            pie_data = top_types.copy()
            if others_count > 0:
                pie_data['Others'] = others_count

            # Create meaningful labels with aircraft type descriptions if available
            labels = []
            for aircraft_type in pie_data.index:
                count = pie_data[aircraft_type]
                if aircraft_type == 'Others':
                    labels.append(f"Others ({count} aircraft)")
                else:
                    # Try to get more descriptive name from aircraft_types data
                    type_info = aircraft_types[aircraft_types['aircraft_type'] == aircraft_type]
                    if not type_info.empty and 'aircraft_type_iata' in type_info.columns:
                        iata_code = type_info['aircraft_type_iata'].iloc[0]
                        labels.append(f"{aircraft_type} ({iata_code})")
                    else:
                        labels.append(f"{aircraft_type}")

            fig_pie = px.pie(
                values=pie_data.values,
                names=labels,
                title="Aircraft Types Distribution (Fleet Composition)",
                hole=0.3,  # Create a donut chart for better visualization
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=11
            )
            fig_pie.update_layout(
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.01,
                    font=dict(size=10)
                )
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            add_explanation("Aircraft Type Distribution Analysis",
                           "Comprehensive fleet composition analysis showing aircraft type distribution as donut chart. Essential for strategic fleet planning, operational requirements assessment, capacity analysis, and understanding fleet diversification. Provides immediate insight into which aircraft types dominate the fleet and potential operational flexibility.",
                           "cleaned_aircraft_registration.csv + cleaned_aircraft_type.csv (merged)",
                           "aircraft_id, aircraft_type, aircraft_type_iata, aircraft_registration",
                           "Merge registrations with aircraft_types on aircraft_id, count per type = COUNT(aircraft_registration) GROUP BY aircraft_type, display top 8 types individually, group remainder as 'Others', calculate percentage distribution",
                           "Types with <8% fleet share grouped as 'Others', donut chart format for optimal readability")

  
def create_business_intelligence_dashboard(processor):
    """Business intelligence dashboard"""
    st.markdown('<h1 class="main-header">Business Intelligence</h1>', unsafe_allow_html=True)

    companies = processor.data['companies']
    airlines = processor.data['airlines']
    schedules = processor.data['schedules']
    delays = processor.calculate_delay_hours()

    if airlines.empty:
        st.warning("No airline data available")
        return

    # Company analysis
    if not companies.empty and not schedules.empty:
        st.markdown('<h2 class="section-header">Company Performance</h2>', unsafe_allow_html=True)

        company_flights = schedules.groupby('company_id').size().reset_index(name='flight_count')
        company_flights = company_flights.merge(
            companies[['id', 'name']],
            left_on='company_id',
            right_on='id',
            how='left'
        ).sort_values('flight_count', ascending=False).head(10)

        fig = px.bar(
            company_flights,
            x='flight_count',
            y='name',
            orientation='h',
            title="Top 10 Companies (Flight Volume)",
            labels={'flight_count': 'Number of Flights', 'name': 'Company'},
            color='flight_count',
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

        add_explanation("Company Performance",
                       "Shows flight volumes per company. Important for partnership evaluation and market analysis.",
                       "cleaned_company.csv, cleaned_schedules after 2024.csv",
                       "company_id, name,",
                       "GROUP BY company_id, COUNT(flight_schedule_id), JOIN with company table for names, ORDER BY flight_count DESC LIMIT 10",
                       "Companies with flight schedules only")

    # Station performance
    if not delays.empty:
        st.markdown('<h2 class="section-header">Station Performance</h2>', unsafe_allow_html=True)

        station_analysis = delays.groupby('arrival_station_id')['delay_duration_hours'].agg(['mean', 'count']).reset_index()
        station_analysis = station_analysis[station_analysis['count'] >= 5]
        station_analysis = station_analysis.sort_values('mean', ascending=False).head(15)

        station_mapping = {
            'AMS': 'Amsterdam', 'LHR': 'London Heathrow', 'CDG': 'Paris CDG',
            'FRA': 'Frankfurt', 'MAD': 'Madrid', 'BCN': 'Barcelona'
        }

        station_analysis['display_name'] = station_analysis['arrival_station_id'].astype(str).map(
            lambda x: station_mapping.get(x, f"Station {x}")
        )

        fig = px.bar(
            station_analysis,
            x='mean',
            y='display_name',
            orientation='h',
            title="Top 15 Stations (Gemiddelde Delay)",
            labels={'mean': 'Gemiddelde Delay (Uren)', 'display_name': 'Station'},
            color='mean',
            color_continuous_scale='reds'
        )
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

        add_explanation("Station Performance",
                       "Identifies stations with the most technical problems. "
                       "Important for ground handling planning and resource allocation.")

def create_enhanced_trend_analysis_dashboard(processor):
    """Trend Analysis Dashboard - KLM Exclusive Analysis"""
    st.markdown('<h1 class="main-header">KLM Trend Analysis</h1>', unsafe_allow_html=True)

    # Filter op KLM data - Enhanced company_id filtering voor correctie analyse
    # Company ID 1 = KLM, want dit is de juiste manier om KLM-specifieke data te filteren
    schedules = processor.data['schedules']
    schedules = schedules[schedules['company_id'] == 1].copy()
    delays = processor.calculate_delay_hours()
    # Filter delays for KLM using the filter_klm_data method
    delays = processor.filter_klm_data(delays)
    delays_merged = processor.merge_with_aircraft_info(delays)

    if schedules.empty:
        st.warning("No KLM schedule data available for trend analysis")
        return

    # Data summary for sidebar
    st.sidebar.markdown(f"**KLM Data Summary:**")
    st.sidebar.metric("KLM Flights Available", f"{len(schedules):,}")
    # Add filters in sidebar
    st.sidebar.markdown("### Custom Time Period Selection")

    if not schedules['scheduled_departure_time'].empty:
        min_date = schedules['scheduled_departure_time'].min().date()
        max_date = schedules['scheduled_departure_time'].max().date()

        # Quick period selection
        st.sidebar.markdown("**Quick Selection:**")
        period_options = [
            "Custom Range",
            "Last 7 Days",
            "Last 30 Days",
            "Last Quarter",
            "Year to Date",
            "Last Year",
            "All Data"
        ]
        selected_period = st.sidebar.selectbox(
            "Select Period:",
            options=period_options,
            index=0
        )

        # Custom date range
        if selected_period == "Custom Range":
            start_date = st.sidebar.date_input(
                "Start Date:",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
            end_date = st.sidebar.date_input(
                "End Date:",
                value=pd.Timestamp(2025, 8, 1).date(),
                min_value=min_date,
                max_value=max_date
            )
        else:
            # Calculate dates based on selection
            today = max_date

            if selected_period == "Last 7 Days":
                start_date = today - pd.Timedelta(days=7)
                end_date = today
            elif selected_period == "Last 30 Days":
                start_date = today - pd.Timedelta(days=30)
                end_date = today
            elif selected_period == "Last Quarter":
                start_date = today - pd.Timedelta(days=90)
                end_date = today
            elif selected_period == "Year to Date":
                start_date = pd.Timestamp(today.year, 1, 1).date()
                end_date = today
            elif selected_period == "Last Year":
                start_date = pd.Timestamp(today.year - 1, 1, 1).date()
                end_date = pd.Timestamp(today.year - 1, 12, 31).date()
            else:  # All Data
                start_date = min_date
                end_date = max_date

        # Convert to datetime for filtering
        start_datetime = pd.Timestamp(start_date)
        end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # Include end date

        # Display selected period
        st.sidebar.markdown(f"**Selected Period:**")
        st.sidebar.markdown(f"{start_date} to {end_date}")

        # Filter data based on selections
        schedules_filtered = schedules[
            (schedules['scheduled_departure_time'] >= start_datetime) &
            (schedules['scheduled_departure_time'] < end_datetime)
        ].copy()

        # Filter delays voor KLM data - merge schedules met delays via flight nummers
        if not delays.empty and not schedules_filtered.empty:
            # Gebruik flight numbers om delays te koppelen aan KLM schedules
            klm_flight_numbers = set()
            for _, schedule in schedules_filtered.iterrows():
                if pd.notna(schedule.get('outbound_flight_number')) and schedule['outbound_flight_number'] != 'Unknown':
                    klm_flight_numbers.add(str(schedule['outbound_flight_number']))
                if pd.notna(schedule.get('correct_flight_number')) and schedule['correct_flight_number'] != 'N/A':
                    klm_flight_numbers.add(str(schedule['correct_flight_number']))

            delays_filtered = delays[
                delays['departure_flight_number'].astype(str).isin(klm_flight_numbers)
            ].copy()
        else:
            delays_filtered = pd.DataFrame()

        # Show data summary
        st.sidebar.markdown(f"**KLM Data Summary:**")
        st.sidebar.metric("KLM Flights in Period", f"{len(schedules_filtered):,}")
        st.sidebar.metric("KLM Delays in Period", f"{len(delays_filtered):,}")

    else:
        min_date = datetime.now().date()
        max_date = datetime.now().date()
        # Add een extra niveau inspringing voor de else tak
        st.sidebar.markdown("### Custom Time Period Selection")

        schedules_filtered = schedules.copy()
        delays_filtered = delays.copy()
        st.sidebar.warning("No KLM date data available for filtering")

        # Set default dates when no date data is available
        start_date = datetime.now().date()
        end_date = datetime.now().date()

    st.markdown('<h2 class="section-header">Flight Trends Over Time</h2>', unsafe_allow_html=True)

    # Daily flight trends (with filtered data)
    if not schedules_filtered.empty:
        schedules_filtered['date'] = schedules_filtered['scheduled_departure_time'].dt.date
        daily_flights = schedules_filtered.groupby('date').size().reset_index(name='flight_count')

        fig = px.line(
            daily_flights,
            x='date',
            y='flight_count',
            title=f"Daily Flight Trends ({start_date} to {end_date})",
            labels={'date': 'Date', 'flight_count': 'Number of Flights'},
            markers=True
        )
        fig.update_traces(line_color='#0066CC', line_width=2)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        add_explanation("KLM Daily Flight Trends",
                       "Time series analysis of KLM flight volumes per day based on scheduled_departure_time. Essential for operational planning, capacity management, and identifying operational peaks and troughs in KLM fleet activity.",
                       "cleaned_schedules after 2024.csv (KLM filtered)",
                       "scheduled_departure_time, company_id",
                       "Daily trend = EXTRACT(date from scheduled_departure_time), COUNT(*) GROUP BY date, ORDER BY date, filtered on company_id = 1",
                       "No filters - only company_id = 1")
    else:
        st.info("No KLM flight data available for selected period")

    # Weekly flight patterns (with filtered data)
    if not schedules_filtered.empty:
        schedules_filtered['week'] = schedules_filtered['scheduled_departure_time'].dt.isocalendar().week
        weekly_flights = schedules_filtered.groupby('week').size().reset_index(name='flight_count')

        fig = px.line(
            weekly_flights,
            x='week',
            y='flight_count',
            title=f"Weekly Flight Patterns ({start_date} to {end_date})",
            labels={'week': 'Week', 'flight_count': 'Number of Flights'},
            markers=True
        )
        fig.update_traces(line_color='#00a651', line_width=2)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        add_explanation("KLM Weekly Flight Patterns",
                       "KLM flight patterns per week based on ISO week numbers (isocalendar().week). Identifies structural weekly operational patterns, crucial for crew planning, maintenance schedules, and fleet efficiency optimization.",
                       "cleaned_schedules after 2024.csv (KLM filtered)",
                       "scheduled_departure_time, company_id",
                       "Weekly analysis = EXTRACT(ISO week from scheduled_departure_time), COUNT(*) GROUP BY week, ORDER BY week, filtered on company_id = 1",
                       "No filters - only company_id = 1")
    else:
        st.info("No weekly KLM data available for selected period")

    # Delay trends if available (with filtered data)
    if not delays_filtered.empty:
        st.markdown('<h2 class="section-header">Delay Trends Over Time</h2>', unsafe_allow_html=True)

        delays_filtered['date'] = delays_filtered['call_time'].dt.date
        daily_delays = delays_filtered.groupby('date')['delay_duration_hours'].mean().reset_index()
        daily_delay_count = delays_filtered.groupby('date').size().reset_index(name='delay_count')

        # Create complete date range to fill missing days with 0 delays
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        complete_delay_df = pd.DataFrame({'date': all_dates.date})

        # Merge with actual delay data to fill missing days with 0
        daily_delay_count = complete_delay_df.merge(
            daily_delay_count, on='date', how='left'
        ).fillna(0)

        # For average delay duration, fill missing days with 0 as well
        daily_delays = complete_delay_df.merge(
            daily_delays, on='date', how='left'
        ).fillna(0)

        # Average delay duration
        fig = px.bar(
            daily_delays,
            x='date',
            y='delay_duration_hours',
            title=f"Average Delay Duration ({start_date} to {end_date})",
            labels={'date': 'Date', 'delay_duration_hours': 'Average Delay (Hours)'},
            color='delay_duration_hours',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        add_explanation("KLM Average Delay Duration",
                       "Bar chart analysis of KLM technical delay duration per day based on call_time and calculated delay_duration_hours. Essential for identifying operational disruption patterns, maintenance planning, and performance indicators.",
                       "cleaned_technical_delay.csv (KLM filtered)",
                       "call_time, delay_duration_hours, technical_delay_id",
                       "Daily average = MEAN(delay_duration_hours) GROUP BY date, filtered on KLM data via departure_flight_number matching, ORDER BY date. Missing days filled with 0 delays.",
                       "KLM technical delays only, filtered on valid duration (>0 hours) and matched with KLM flight numbers")

        # Number of delays
        fig = px.bar(
            daily_delay_count,
            x='date',
            y='delay_count',
            title=f"Number of KLM Delays ({start_date} to {end_date})",
            labels={'date': 'Date', 'delay_count': 'Number of Delays'},
            color='delay_count',
            color_continuous_scale='Oranges'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Box plots for trends by day of week (with filtered data)
        st.markdown('<h2 class="section-header">Delay Distribution Analysis</h2>', unsafe_allow_html=True)

        delays_filtered['day_of_week'] = delays_filtered['call_time'].dt.day_name()
        delays_filtered['month'] = delays_filtered['call_time'].dt.month_name()

        # Box plot by day of week - ensure correct order
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_delays = delays_filtered.groupby('day_of_week')['delay_duration_hours'].apply(list).reset_index()
        weekday_delays['day_of_week'] = pd.Categorical(weekday_delays['day_of_week'], categories=weekday_order, ordered=True)
        weekday_delays = weekday_delays.sort_values('day_of_week')

        fig = go.Figure()
        for day in weekday_order:
            if day in weekday_delays['day_of_week'].values:
                day_data = weekday_delays[weekday_delays['day_of_week'] == day]['delay_duration_hours'].iloc[0]
                if len(day_data) >= 3:
                    fig.add_trace(go.Box(
                        y=day_data,
                        name=day,
                        boxpoints='outliers'
                    ))

        fig.update_layout(
            title=f"Delay Distribution by Day ({start_date.year})",
            yaxis_title="Delay (Hours)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Box plot explanation
        add_explanation("How to Read the Box Plot",
                       "A box plot displays the statistical distribution of delay durations, helping identify patterns, variability, and outliers in KLM's technical delay data. This visualization is essential for understanding operational consistency and planning resources effectively.<br><br><strong>Box Plot Components Explained:</strong><br>• <strong>Min (Minimum):</strong> Shortest recorded delay duration<br>• <strong>Q1 (First Quartile):</strong> 25% of delays are shorter than this value<br>• <strong>Median:</strong> Middle value where 50% of delays are shorter and 50% are longer<br>• <strong>Q3 (Third Quartile):</strong> 75% of delays are shorter than this value<br>• <strong>Max (Maximum):</strong> Longest recorded delay duration<br>• <strong>Lower/Upper Fence:</strong> Normal range boundaries (values beyond these are statistical outliers)<br><br><strong>Why we use box plots:</strong><br>• Identify delay patterns across different days or months<br>• Detect operational consistency (smaller box = more predictable delays)<br>• Spot outliers that may require special investigation<br>• Compare delay distributions for better resource planning")

        add_explanation("KLM Delay Distribution by Day of Week",
                       "Statistical distribution analysis of KLM technical delay duration per weekday based on box plot metrics. Identifies seasonal and operational patterns in maintenance impact, crucial for crew planning and resource allocation.",
                       "cleaned_technical_delay.csv (KLM filtered)",
                       "call_time, delay_duration_hours, technical_delay_id",
                       "Box plot analysis = STATISTICAL BOX PLOT (quartiles, median, outliers) GROUP BY day_of_week, filtered on KLM data, minimum 3 data points per day for statistical validity",
                       "KLM technical delays only, box plot visualization for daily variability analysis")

        # Box plot by month - show if date range spans multiple months, ensure correct order
        date_range_days = (end_date - start_date).days
        if date_range_days >= 30:  # Show monthly comparison if range is 30+ days
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_delays = delays_filtered.groupby('month')['delay_duration_hours'].apply(list).reset_index()
            monthly_delays['month'] = pd.Categorical(monthly_delays['month'], categories=month_order, ordered=True)
            monthly_delays = monthly_delays.sort_values('month')

            fig = go.Figure()
            for month in month_order:
                if month in monthly_delays['month'].values:
                    month_data = monthly_delays[monthly_delays['month'] == month]['delay_duration_hours'].iloc[0]
                    if len(month_data) >= 3:
                        fig.add_trace(go.Box(
                            y=month_data,
                            name=month,
                            boxpoints='outliers'
                        ))

            fig.update_layout(
                title=f"KLM Delay Distribution by Month ({start_date.year})",
                yaxis_title="Delay (Hours)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            add_explanation("KLM Monthly Delay Distribution",
                           "Monthly comparative box plot analysis of KLM technical delay patterns. Essential for seasonal comparison, maintenance planning, and identification of systematic operational challenges.",
                           "cleaned_technical_delay.csv (KLM filtered)",
                           "call_time, delay_duration_hours, technical_delay_id",
                           "Monthly box plot = STATISTICAL BOX PLOT PER MONTH, minimum 3 incidents per month, comparative monthly distributions for seasonal patterns",
                           "KLM delays only, monthly comparison (>30 days period)")
        else:
            st.info(f"Select date range of 30+ days to see monthly comparison.")
    else:
        st.info("No KLM delay data available for selected period")


def create_defect_action_dashboard(processor):
    """Defect & Action Analysis Dashboard - Enhanced from original"""
    st.markdown('<h1 class="main-header">Defect & Action Analysis</h1>', unsafe_allow_html=True)

    delays = processor.calculate_delay_hours()
    # Filter delays for KLM
    delays = processor.filter_klm_data(delays)
    actions = processor.data['actions']  # Already KLM filtered

    if delays.empty:
        st.warning("No delay data available for defect analysis")
        return

    st.markdown('<h2 class="section-header">Most Common Defect Categories</h2>', unsafe_allow_html=True)

    # Most frequent delay codes
    defect_codes = delays['delay_code_description1'].value_counts().head(15)
    defect_df = pd.DataFrame({'defect_code': defect_codes.index, 'count': defect_codes.values})

    fig = px.bar(
        defect_df,
        x='count',
        y='defect_code',
        orientation='h',
        title="Top 15 Defect Categories",
        labels={'count': 'Number of Delays', 'defect_code': 'Defect Category'},
        color='count',
        color_continuous_scale='oranges'
    )
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    add_explanation("Most Frequent Defect Categories",
                   "Analysis of most common defect types based on delay_code_description1. Frequency analysis identifies systematic technical problems that repeatedly occur in the fleet, crucial for preventive maintenance and training programs.",
                   "cleaned_technical_delay.csv",
                   "delay_code_description1, technical_delay_id, delay_duration_hours",
                   "Frequency analysis = COUNT(delay_code_description1) GROUP BY defect category, sorted by highest frequency, TOP 15 results",
                   "Technical delays with valid delay_code_description1, limited to top 15 most common defects")

    # Average delay by defect category
    defect_avg = delays.groupby('delay_code_description1')['delay_duration_hours'].agg(['mean', 'count']).reset_index()
    defect_avg = defect_avg[defect_avg['count'] >= 3]  # Filter categories with enough data
    defect_avg = defect_avg.sort_values('mean', ascending=False).head(10)

    fig = px.bar(
        defect_avg,
        x='mean',
        y='delay_code_description1',
        orientation='h',
        title="Average Delay by Defect Category",
        labels={'mean': 'Average Delay (Hours)', 'delay_code_description1': 'Defect Category'},
        color='mean',
        color_continuous_scale='reds'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    add_explanation("Average Delay by Defect Category",
                   "Weighted average analysis per defect type based on delay_code_description1 and actual delay_duration_hours. Identifies defect categories with the highest operational impact, essential for maintenance resource prioritization and critical path analysis.",
                   "cleaned_technical_delay.csv",
                   "delay_code_description1, delay_duration_hours, technical_delay_id",
                   "Average calculation = MEAN(delay_duration_hours) GROUP BY delay_code_description1, filter on categories with ≥3 incidents for statistical reliability, sort by highest average, TOP 10 results",
                   "Technical delays with valid duration (>0 hours), minimum 3 incidents per category for statistical validity, top 10 by average impact")

    # ATA Chapter Analysis (similar to Defect Category Analysis)
    st.markdown('<h2 class="section-header">Defect Category Analysis</h2>', unsafe_allow_html=True)

    # Top defect categories by average delay
    defect_analysis = delays.groupby('ata_chapter_name')['delay_duration_hours'].agg(['mean', 'count']).reset_index()
    defect_analysis = defect_analysis[defect_analysis['count'] >= 3]  # Filter categories with enough data
    defect_analysis = defect_analysis.sort_values('mean', ascending=False).head(10)

    fig = px.bar(
        defect_analysis,
        x='mean',
        y='ata_chapter_name',
        orientation='h',
        title="Top 10 Defect Categories by Average Delay",
        labels={'mean': 'Average Delay (Hours)', 'ata_chapter_name': 'Defect Category'},
        color='mean',
        color_continuous_scale='oranges'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    add_explanation("Defect Categories - Average Impact",
                   "ATA chapter based impact analysis with weighted averages per defect category (ata_chapter_name). Combines standardization (ATA chapters) with operational impact measurement, crucial for reliable component reliability and maintenance planning.",
                   "cleaned_technical_delay.csv",
                   "ata_chapter_name, delay_duration_hours, technical_delay_id",
                   "ATA Chapter analysis = MEAN(delay_duration_hours) GROUP BY ata_chapter_name, filter on chapters with ≥3 incidents, sort by highest average, TOP 10 results",
                   "Technical delays with valid ATA chapter code, minimum 3 incidents per chapter for statistical relevance, top 10 by operational impact")

    # Most frequent defect categories
    defect_freq = delays['ata_chapter_name'].value_counts().head(10)

    defect_freq_df = pd.DataFrame({'defect_category': defect_freq.index, 'count': defect_freq.values})
    fig = px.bar(
        defect_freq_df,
        x='count',
        y='defect_category',
        orientation='h',
        title="Top 10 Most Frequent Defect Categories",
        labels={'count': 'Number of Delays', 'defect_category': 'Defect Category'},
        color='count',
        color_continuous_scale='purples'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    add_explanation("Defect Categories - Frequency",
                   "ATA chapter frequency analysis based on systematic defect classification (ata_chapter_name). Identifies structural patterns in technical failures and component reliability, fundamental for predictive maintenance and quality control programs.",
                   "cleaned_technical_delay.csv",
                   "ata_chapter_name, technical_delay_id, delay_duration_hours",
                   "Frequency analysis = COUNT(technical_delay_id) GROUP BY ata_chapter_name, sorted by highest frequency, TOP 10 results",
                   "Technical delays with valid ATA chapter classification, limited to top 10 most common ATA chapters for focus on structural reliability problems")

    # Action analysis if available
    if not actions.empty:
        st.markdown('<h2 class="section-header">Action Analysis</h2>', unsafe_allow_html=True)

        # Actions per delay analysis
        actions_per_delay = actions.groupby('technical_delay_id').size().reset_index(name='action_count')

        # Calculate metrics
        total_delays_with_actions = len(actions_per_delay)
        rectification_count = actions_per_delay[actions_per_delay['action_count'] == 1].shape[0]
        troubleshooting_count = actions_per_delay[actions_per_delay['action_count'] > 1].shape[0]
        avg_actions = actions_per_delay['action_count'].mean()

        # Display metrics in four columns for better spacing
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{rectification_count:,}</div>
                <div class="metric-label">Rectification</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem;">
                    Single action delays<br>
                    ({(rectification_count/total_delays_with_actions)*100:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{troubleshooting_count:,}</div>
                <div class="metric-label">Troubleshooting</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem;">
                    Multiple action delays<br>
                    ({(troubleshooting_count/total_delays_with_actions)*100:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{avg_actions:.1f}</div>
                <div class="metric-label">Avg Actions per Delay</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{total_delays_with_actions:,}</div>
                <div class="metric-label">Total Delays with Actions</div>
            </div>
            """, unsafe_allow_html=True)

        add_explanation("Action Types Analysis",
                       "Quantitative analysis of maintenance actions per technical delay (technical_delay_id). Classifies the complexity of repair procedures - rectification (single action) versus troubleshooting (multiple actions). Essential for resource planning, technical staffing, and maintenance process effectiveness measurement.",
                       "cleaned_technical_delay.csv + cleaned_technical_delay_action.csv (joined on technical_delay_id)",
                       "technical_delay_id, action_id, action_count (calculated)",
                       "Action analysis = COUNT(action_id) GROUP BY technical_delay_id, calculation percentages: rectification = WHERE action_count = 1, troubleshooting = WHERE action_count > 1, average = MEAN(action_count)",
                       "Technical delays with registered actions, classification in rectification (1 action) and troubleshooting (>1 action)")

def create_company_airline_dashboard(processor):
    """Company & Airline Analysis Dashboard - Enhanced from original"""
    st.markdown('<h1 class="main-header">Company & Airline Analysis</h1>', unsafe_allow_html=True)

    companies = processor.data['companies']
    airlines = processor.data['airlines']
    schedules = processor.data['schedules']

    if airlines.empty:
        st.warning("No airline data available")
        return

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_companies = len(companies) if not companies.empty else 0
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{total_companies:,}</div>
            <div class="metric-label">Total Companies</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{len(airlines):,}</div>
            <div class="metric-label">Total Airlines</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        total_flights = len(schedules) if not schedules.empty else 0
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{total_flights:,}</div>
            <div class="metric-label">Total Flights</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        unique_aircraft = schedules['aircraft_id'].nunique() if not schedules.empty else 0
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{unique_aircraft:,}</div>
            <div class="metric-label">Aircraft in Schedule</div>
        </div>
        """, unsafe_allow_html=True)

    add_explanation("Company & Airline Overview",
                   "High-level overview of all companies, airlines, and flights in the dataset. Provides context for the detailed analyses that follow.",
                   "cleaned_company.csv, cleaned_airline.csv, cleaned_schedules after 2024.csv",
                   "company_id, id (companies), airline_id (airlines), aircraft_id (schedules)",
                   "Total companies = COUNT(all rows in companies), Total airlines = COUNT(all rows in airlines), Total flights = COUNT(all rows in schedules), Aircraft in schedule = COUNT(DISTINCT aircraft_id)",
                   "No filters - all data shown")

  
    # Airlines per company
    if 'company_id' in airlines.columns and not companies.empty:
        company_airlines = airlines.groupby('company_id').size().reset_index(name='airline_count')
        company_airlines = company_airlines.merge(
            companies[['id', 'name']],
            left_on='company_id',
            right_on='id',
            how='left'
        ).sort_values('airline_count', ascending=False).head(10)

        fig = px.bar(
            company_airlines,
            x='airline_count',
            y='name',
            orientation='h',
            title="Airlines per Company (Top 10)",
            labels={'airline_count': 'Aantal Airlines', 'name': 'Company'},
            color='airline_count',
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

        add_explanation("Airlines per Company",
                       "Shows how many airlines each company owns. Important for partnership evaluation and market analysis. Helps identify major aviation companies and their market share.",
                       "cleaned_airline.csv + cleaned_company.csv (merged)",
                       "company_id (airlines), id (companies), name (companies), airline_id (airlines)",
                       "Number of airlines per company = COUNT(airline_id) GROUP BY company_id, merge with company names on id, sort by highest count, limited to top 10",
                       "Top 10 companies with most airlines")

  
    # Airline Performance Section
    if not schedules.empty:
        st.markdown('<h2 class="section-header">Airline Performance Analysis</h2>', unsafe_allow_html=True)

        # Flights per airline
        airline_flights = schedules.groupby('airline_id').size().reset_index(name='flight_count')
        airline_flights = airline_flights.merge(airlines, on='airline_id', how='left')
        airline_flights = airline_flights.sort_values('flight_count', ascending=False).head(15)

        if not airline_flights.empty and 'airline_name' in airline_flights.columns:
            fig = px.bar(
                airline_flights,
                x='flight_count',
                y='airline_name',
                orientation='h',
                title="Top 15 Airlines by Flight Volume",
                labels={'flight_count': 'Number of Flights', 'airline_name': 'Airline'},
                color='flight_count',
                color_continuous_scale='plasma'
            )
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)

            add_explanation("Airline Flight Volume",
                           "Shows busiest airlines by number of flights. "
                           "Important for market analysis and capacity planning.",
                           "cleaned_schedules after 2024.csv, cleaned_airline.csv",
                           "airline_id, airline_name",
                           "COUNT(flight_schedule_id) GROUP BY airline_id, JOIN with airline table for names, ORDER BY flight_count DESC LIMIT 15",
                           "Airlines with scheduled flights only")

        # Airlines by IATA code distribution
        if 'iata_code' in airlines.columns:
            iata_distribution = airlines['iata_code'].value_counts().head(20)

            fig = px.pie(
                values=iata_distribution.values,
                names=iata_distribution.index,
                title="Airlines by IATA Code (Top 20)"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            add_explanation("IATA Code Distribution",
                           "Shows distribution of airlines per IATA code. "
                           "Important for route planning and codeshare analysis.",
                           "cleaned_airline.csv",
                           "iata_code, airline_id",
                           "COUNT(airline_id) GROUP BY iata_code, ORDER BY count DESC LIMIT 20",
                           "Top 20 IATA codes by airline count")


  

def create_upload_interface():
    """Create file upload interface with data cleaning capabilities"""
    st.markdown('<h1 class="main-header">📁 KLM Data Upload & Cleaning</h1>', unsafe_allow_html=True)

    # File upload section
    st.markdown('<h2 class="section-header">Upload CSV Files</h2>', unsafe_allow_html=True)

    # Define expected files
    expected_files = {
        'schedules': ['schedules', 'flight_schedule', 'roster', 'cleaned_schedules'],
        'delays': ['delay', 'technical_delay', 'maintenance_delay', 'cleaned_technical_delay'],
        'actions': ['action', 'technical_delay_action', 'maintenance_action', 'cleaned_technical_delay_action'],
        'airlines': ['airline', 'carrier', 'cleaned_airline'],
        'aircraft_types': ['aircraft_type', 'aircraft', 'fleet', 'cleaned_aircraft_type'],
        'registrations': ['registration', 'aircraft_registration', 'tail_number', 'cleaned_aircraft_registration'],
        'companies': ['company', 'operator', 'cleaned_company'],
        'clusters': ['cluster', 'aircraft_cluster', 'type_cluster', 'cleaned_aircraft_type_cluster']
    }

    # Create upload interface
    uploaded_files = st.file_uploader(
        "Upload CSV files (you can select multiple)",
        type=['csv'],
        accept_multiple_files=True,
        key="klm_csv_files_uploader",
        help="Upload your raw CSV files. The system will automatically detect file types and clean the data."
    )

    # Initialize session state for uploaded data
    if 'original_data' not in st.session_state:
        st.session_state.original_data = {}
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = {}
    if 'cleaning_stats' not in st.session_state:
        st.session_state.cleaning_stats = {}

    if uploaded_files:
        st.sidebar.markdown(f'<div class="info-box">Uploaded {len(uploaded_files)} files</div>', unsafe_allow_html=True)

        # Load files and detect types
        successfully_loaded = 0
        failed_files = []
        detected_files = {}

        for file in uploaded_files:
            try:
                file.seek(0)
                df = pd.read_csv(file)

                if df.empty:
                    st.sidebar.warning(f"⚠️ {file.name} is empty")
                    failed_files.append(file.name)
                else:
                    # Detect file type based on filename and content
                    file_type = detect_file_type(file.name, df, expected_files)
                    detected_files[file_type] = df
                    st.session_state.original_data[file_type] = df
                    st.sidebar.success(f"✓ {file.name} loaded as {file_type} ({df.shape[0]} rows, {df.shape[1]} columns)")
                    successfully_loaded += 1

            except Exception as e:
                st.sidebar.error(f"✗ {file.name}: {str(e)}")
                failed_files.append(file.name)

        # Show upload summary
        if successfully_loaded > 0:
            st.sidebar.info(f"📊 Successfully processed {successfully_loaded} files")

        if failed_files:
            st.sidebar.error(f"❌ Failed to process {len(failed_files)} files: {', '.join(failed_files)}")

        # Show data preview and cleaning options
        if detected_files:
            st.markdown('<h2 class="section-header">Data Preview & Cleaning Options</h2>', unsafe_allow_html=True)

            # Show file summary
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Files Detected", len(detected_files))
            with col2:
                total_rows = sum(df.shape[0] for df in detected_files.values())
                st.metric("Total Rows", f"{total_rows:,}")
            with col3:
                total_cols = sum(df.shape[1] for df in detected_files.values())
                st.metric("Total Columns", total_cols)
            with col4:
                null_count = sum(df.isnull().sum().sum() for df in detected_files.values())
                st.metric("Null Values", f"{null_count:,}")

            # File preview sections
            for file_type, df in detected_files.items():
                with st.expander(f"{file_type.title()} Data ({df.shape[0]} rows, {df.shape[1]} columns)", expanded=False):
                    st.dataframe(df.head())

                    # Basic quality metrics
                    null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                    duplicate_count = df.duplicated().sum()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Null %", f"{null_percentage:.1f}%")
                    with col2:
                        st.metric("Duplicates", f"{duplicate_count:,}")
                    with col3:
                        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

            # Clean data automatically and proceed to dashboard
            st.markdown('<h2 class="section-header">Processing Data</h2>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Process & Go to Dashboard", type="primary", use_container_width=True):
                    with st.spinner("Cleaning and processing data..."):
                        # Clean all uploaded data
                        cleaner = DataCleaner()
                        cleaner.original_data = detected_files

                        for file_type, df in detected_files.items():
                            cleaned_df, cleaning_log = cleaner.clean_dataframe(file_type, df)
                            st.session_state.cleaned_data[file_type] = cleaned_df

                            # Store cleaning stats
                            original_stats = cleaner.analyze_data_quality(f"original_{file_type}", df)
                            cleaned_stats = cleaner.analyze_data_quality(f"cleaned_{file_type}", cleaned_df)

                            st.session_state.cleaning_stats[file_type] = {
                                'original': original_stats,
                                'cleaned': cleaned_stats,
                                'cleaning_log': cleaning_log
                            }

                    st.success("✅ Data processed successfully! Loading dashboard...")
                    st.rerun()

            with col2:
                if st.button("Clear All Data", use_container_width=True):
                    keys_to_clear = ['original_data', 'cleaned_data', 'cleaning_stats', 'type_suggestions']
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

            # Processing information
            st.info("""
            **Automatic Processing:**

            When you click "Process & Go to Dashboard", the system will:
            - Clean all uploaded data automatically
            - Remove NULL values and fix data types
            - Optimize data for analysis
            - Load the KLM dashboard with your cleaned data

            The entire process typically takes 10-30 seconds depending on data size.
            """)

    else:
        # Instructions when no files uploaded
        st.markdown("""
        <div class="info-box">
            <h3>Welcome to KLM Operations Hub</h3>
            <p>Upload your raw CSV files to get started with the KLM dashboard:</p>
            <ul>
                <li>Upload your raw CSV files (schedules, delays, actions, airlines, etc.)</li>
                <li>The system will automatically detect file types and clean the data</li>
                <li>View interactive KLM dashboard with visualizations</li>
                <li>Get real-time analysis and operational insights</li>
            </ul>
            <p><strong>Upload your CSV files above to begin!</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Expected files information
        st.markdown('<h2 class="section-header">Expected CSV Files</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>Schedule Data</h4>
                <p><strong>File names containing:</strong> schedule, flight, roster</p>
                <p>Flight schedules, departure/arrival times, aircraft assignments</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
                <h4>Delay Data</h4>
                <p><strong>File names containing:</strong> delay, technical, maintenance</p>
                <p>Technical delays, maintenance issues, ATA chapters</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
                <h4>Action Data</h4>
                <p><strong>File names containing:</strong> action, technical_delay_action</p>
                <p>Maintenance actions, repair procedures, troubleshooting steps</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>Airline & Aircraft Data</h4>
                <p><strong>File names containing:</strong> airline, aircraft, registration</p>
                <p>Airline information, aircraft types, registration details</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
                <h4>Company Data</h4>
                <p><strong>File names containing:</strong> company, operator</p>
                <p>Company information, operator details</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
                <h4>Cluster Data</h4>
                <p><strong>File names containing:</strong> cluster, group, category</p>
                <p>Aircraft type clusters, group classifications</p>
            </div>
            """, unsafe_allow_html=True)

def detect_file_type(filename, df, expected_files):
    """Detect file type based on filename and content"""
    filename_lower = filename.lower()
    columns_lower = [col.lower() for col in df.columns]

    # Score each file type based on filename and column matches
    scores = {}

    for file_type, keywords in expected_files.items():
        score = 0

        # Check filename matches
        for keyword in keywords:
            if keyword in filename_lower:
                score += 3  # Higher weight for filename matches

        # Check column matches based on file type
        if file_type == 'schedules':
            schedule_keywords = ['flight', 'schedule', 'departure', 'arrival', 'airline_id', 'aircraft_id']
            for keyword in schedule_keywords:
                if keyword in ' '.join(columns_lower):
                    score += 2

        elif file_type == 'delays':
            delay_keywords = ['delay', 'technical', 'maintenance', 'call_time', 'end_of_work', 'ata_chapter']
            for keyword in delay_keywords:
                if keyword in ' '.join(columns_lower):
                    score += 2

        elif file_type == 'actions':
            action_keywords = ['action', 'technical_delay', 'maintenance_action', 'description']
            for keyword in action_keywords:
                if keyword in ' '.join(columns_lower):
                    score += 2

        elif file_type == 'airlines':
            airline_keywords = ['airline', 'carrier', 'iata', 'icao', 'airline_name']
            for keyword in airline_keywords:
                if keyword in ' '.join(columns_lower):
                    score += 2

        elif file_type == 'aircraft_types':
            aircraft_keywords = ['aircraft_type', 'engine_count', 'manufacturer', 'model']
            for keyword in aircraft_keywords:
                if keyword in ' '.join(columns_lower):
                    score += 2

        elif file_type == 'registrations':
            reg_keywords = ['registration', 'tail_number', 'aircraft_registration']
            for keyword in reg_keywords:
                if keyword in ' '.join(columns_lower):
                    score += 2

        elif file_type == 'companies':
            company_keywords = ['company', 'operator', 'organization']
            for keyword in company_keywords:
                if keyword in ' '.join(columns_lower):
                    score += 2

        elif file_type == 'clusters':
            cluster_keywords = ['cluster', 'group', 'category', 'aircraft_cluster']
            for keyword in cluster_keywords:
                if keyword in ' '.join(columns_lower):
                    score += 2

        scores[file_type] = score

    # Return the file type with the highest score
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    else:
        # Fallback to generic naming based on content
        if 'flight' in filename_lower or 'schedule' in filename_lower:
            return 'schedules'
        elif 'delay' in filename_lower:
            return 'delays'
        elif 'action' in filename_lower:
            return 'actions'
        elif 'airline' in filename_lower:
            return 'airlines'
        else:
            return 'unknown'

def main():
    """Main application with file upload and navigation"""

    # Header
    st.markdown('<h1 class="main-header">KLM Operations Hub</h1>', unsafe_allow_html=True)

    # Check if data has been uploaded and cleaned
    if 'cleaned_data' not in st.session_state or not st.session_state.cleaned_data:
        # Show upload interface if no data available
        create_upload_interface()
        return

    # Initialize processor with uploaded data
    if 'processor' not in st.session_state:
        st.session_state.processor = KLMDataProcessor(use_uploaded_data=True)
        st.session_state.processor.load_all_data()

    processor = st.session_state.processor

    # Dashboard navigation - moved to top
    st.sidebar.markdown('<h2 style="color: #0066CC;">Dashboard Pages</h2>', unsafe_allow_html=True)

    page = st.sidebar.selectbox(
        "Select Dashboard:",
        [
            "Technical Performance & Defect Analysis",
            "Operations Analytics",
            "Company & Airline Analysis",
            "Trend Analysis"
        ]
    )

    # Add separator
    st.sidebar.markdown("---")

    # Data Status - moved to bottom
    st.sidebar.markdown('<h2 style="color: #0066CC;">Data Status</h2>', unsafe_allow_html=True)

    # Show data status in sidebar
    data_status = processor.get_data_status()
    for data_type, status in data_status.items():
        status_icon = "✅" if "Not available" not in status else "❌"
        st.sidebar.write(f"{status_icon} **{data_type.title()}:** {status}")

    # Upload new files option
    st.sidebar.markdown("---")
    if st.sidebar.button("📁 Upload New Files", use_container_width=True):
        # Clear current data and return to upload interface
        keys_to_clear = ['original_data', 'cleaned_data', 'cleaning_stats', 'type_suggestions', 'processor']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # Check if data is available for dashboard
    if not has_sufficient_data(processor):
        st.error("⚠️ Insufficient data available for dashboard. Please upload additional CSV files.")
        st.info("Required data files: schedules, delays, actions, airlines, aircraft_types, registrations, companies")
        return

    # Load selected page
    if page == "Technical Performance & Defect Analysis":
        create_technical_performance_dashboard(processor)
    elif page == "Operations Analytics":
        create_operations_fleet_dashboard(processor)
    elif page == "Company & Airline Analysis":
        create_company_airline_dashboard(processor)
    elif page == "Trend Analysis":
        create_enhanced_trend_analysis_dashboard(processor)

def has_sufficient_data(processor):
    """Check if there's sufficient data for dashboard functionality"""
    required_files = ['schedules', 'delays', 'actions', 'airlines', 'aircraft_types', 'registrations', 'companies']
    available_count = 0

    for file_type in required_files:
        if file_type in processor.data and not processor.data[file_type].empty:
            available_count += 1

    # Require at least 3 essential files to show dashboard
    essential_files = ['schedules', 'delays', 'airlines']
    essential_available = sum(1 for file_type in essential_files
                            if file_type in processor.data and not processor.data[file_type].empty)

    return essential_available >= 2 or available_count >= 3

def safe_data_operation(operation_func, default_value=None, error_message="Operation failed"):
    """Safely execute data operations with error handling"""
    try:
        return operation_func()
    except Exception as e:
        st.error(f"⚠️ {error_message}: {str(e)}")
        return default_value

if __name__ == "__main__":
    main()