import sys
import pandas as pd
from pathlib import Path
#sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data_processor import DataProcessor
from src.predictor import Predictor
import streamlit as st
import plotly.express as px
import numpy as np
import json
import statsmodels

st.set_page_config(page_title="Weather Dashboard", layout="wide")

DAY_COLORS = {"Sunny": "yellow", "Cloudy": "grey", "Rainy": "blue", "Hot": "red", "Cold": "white"}

@st.cache_resource
def load_processor():
    return DataProcessor()
@st.cache_resource
def load_predictor():
    return Predictor()

data_processor = load_processor()
df = data_processor.df
raw_df = data_processor.raw_df

st.title("Weather Dashboard")

overview, explore, predict, performance = st.tabs(["Data Overview", "Exploration", "Prediction", "Model Performance"])

with overview:
    st.subheader("Dataset Size")
    rows_card, cols_card, cells_card = st.columns(3)
    with rows_card, st.container(border=True):
        st.metric("Rows", f"{len(raw_df):,}")
    with cols_card, st.container(border=True):
        st.metric("Columns", len(raw_df.columns))
    with cells_card, st.container(border=True):
        st.metric("Cells", f"{raw_df.size:,}")

    st.subheader("Preview Table")
    st.dataframe(raw_df.head(20), use_container_width=True, hide_index=True)

    st.subheader("Summary Statistics")
    st.dataframe(raw_df.describe().round(2), use_container_width=True)

with explore:
    filtered_df = raw_df

    with st.container(border=True):
        day_types = sorted(raw_df['day_type'].unique())
        chosen_day_types = st.multiselect("Day types", day_types, default=day_types)
        filtered_df = filtered_df[filtered_df['day_type'].isin(chosen_day_types)]
        features = raw_df.columns[:-1]
        slider_cols = st.columns(len(features))
        for col, attrib_name in zip(slider_cols, features):
            raw_min = raw_df[attrib_name].min()
            raw_max = raw_df[attrib_name].max()
            low, high = col.slider(attrib_name.replace('_', ' ').capitalize(), min_value=raw_min, max_value=raw_max, value=(raw_min, raw_max))
            filtered_df = filtered_df[filtered_df[attrib_name].between(low, high)]

    possible_features = list(raw_df.columns[1:-1])
    (left, right) = st.columns(2)

    # Bar chart
    with left, st.container(border=True):
        bar_df = filtered_df['day_type'].value_counts().reset_index()
        barfig = px.bar(bar_df, x='day_type', y='count', color='day_type',
                        color_discrete_map=DAY_COLORS, text_auto=True,
                        title="Number of Days per Day Type",
                        labels={'day_type': 'Day type', 'count': 'Count'})
        barfig.update_traces(textposition='outside', cliponaxis=False)
        barfig.update_layout(showlegend=False)
        st.plotly_chart(barfig, use_container_width=True)
        st.caption("The number of days in the dataset that correspond to a particular day type.")

    # Line chart
    with right, st.container(border=True):
        line_yaxis = st.selectbox("Select Feature", possible_features)
        line_df = filtered_df[['day_of_year', line_yaxis]].groupby('day_of_year').mean().reset_index()
        linefig = px.line(line_df, x='day_of_year', y=line_yaxis,title=f"Average {line_yaxis} over day_over_year",labels={'day_of_year': 'Day of year'})
        st.plotly_chart(linefig, use_container_width=True)
        st.caption(f"A line plot showing how {line_yaxis} changes over a year.")

    # Scatter
    with st.container(border=True):
        sx, sy = st.columns(2)
        xscatter = sx.selectbox("X axis", possible_features, index=0)
        yscatter = sy.selectbox("Y axis", possible_features, index=1)
        scatterfig = px.scatter(filtered_df, trendline='ols', trendline_scope="overall", x=xscatter, y=yscatter, color='day_type',color_discrete_map=DAY_COLORS,title=f"{yscatter} vs {xscatter}",labels={'day_type': 'Day type'})
        st.plotly_chart(scatterfig, use_container_width=True)
        st.caption(f"A scatter plot comparing {xscatter} and {yscatter} over time.")

with predict:
    st.subheader("Predict the Day Type")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        inputs = {
            'day_of_year': c1.number_input("Day of year", 1, 365, 180),
            'day_temp':    c2.slider("Temperature (°C)", 5.0, 48.0, 25.0),
            'humidity':    c3.slider("Humidity (%)", 10.0, 100.0, 60.0),
            'cloud_cover': c1.slider("Cloud cover (%)", 0.0, 100.0, 30.0),
            'uv_index':    c2.slider("UV index", 0.0, 11.0, 5.0),
        }
    if st.button("Predict"):
        label, probabilities = load_predictor().predict(inputs)
        with st.container(border=True):
            font_color = DAY_COLORS[label]
            st.markdown(f'Predicted day type: <span style="color: {font_color}; font-weight: bold;">{label}</span>',unsafe_allow_html=True)
            prob_df = pd.DataFrame({'Day Type':probabilities.keys(), 'Probability':np.array(list(probabilities.values()))*100})
            st.dataframe(prob_df.round(3), hide_index=True)
            piefig = px.pie(prob_df.round(3), title="Probability of Day Type", values='Probability', names='Day Type', color='Day Type', color_discrete_map=DAY_COLORS)
            st.plotly_chart(piefig, use_container_width=True)
            st.caption("The probability of the conditions provided resulting in a particular day type, estimated by the best model.")

with performance:
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)

    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)

    st.subheader("Best Model")
    st.table(pd.DataFrame([metrics])[['model', 'name', 'train_time', 'f1', 'precision', 'accuracy', 'recall']].round(3))

    st.subheader("Model Comparison")
    comparison = pd.read_csv('models/model_comparison.csv')
    comparison = comparison.sort_values('validation_f1', ascending=False)
    st.dataframe(comparison.round(3), hide_index=True, use_container_width=True)

    with st.container(border=True):
        metric_cols = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        long_df = comparison.melt(id_vars='model', value_vars=metric_cols, var_name='metric', value_name='score')
        compfig = px.bar(long_df, x='metric', y='score', color='model', barmode='group', text_auto='.3f', range_y=[0, 1.05], title="Test Metrics by Model")
        compfig.update_traces(textposition='outside')
        st.plotly_chart(compfig, use_container_width=True)
        st.caption("The top-performing models of each classification family are shown.")

    left, right = st.columns(2)

    with left, st.container(border=True):
        cmfig = px.imshow(
            metrics['confusion_matrix'],
            title="Confusion Matrix",
                          x=metrics['class_labels'], y=metrics['class_labels'],
                          text_auto=True, color_continuous_scale='Blues',
                          labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Count'})
        cmfig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(cmfig, use_container_width=True)
        st.caption("Rows are the actual value; the columns are what the best model predicted they would be.")

    with right, st.container(border=True):
        fi_df = pd.Series(metrics['feature_ranking']).sort_values()
        fifig = px.bar(title="Feature Importances", x=fi_df.values, y=fi_df.index, orientation='h', text_auto='.3f',
                       labels={'x': 'Importance', 'y': 'Feature'})
        fifig.update_traces(textposition='outside', cliponaxis=False)
        st.plotly_chart(fifig, use_container_width=True)
        st.caption("This is how the best model ranked the importance of the features in making its predictions.")