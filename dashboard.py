import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data_processor import DataProcessor
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Weather Dashboard", layout="wide")

DAY_COLORS = {"Sunny": "yellow", "Cloudy": "grey", "Rainy": "blue", "Hot": "red", "Cold": "white"}

@st.cache_resource
def load_processor():
    return DataProcessor()

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
        barfig = px.bar(bar_df, x='day_type', y='count', color='day_type', color_discrete_map=DAY_COLORS, title="Number of Days per Day Type", labels={'day_type': 'Day type', 'count': 'Count'})
        barfig.update_layout(showlegend=False)
        st.plotly_chart(barfig, use_container_width=True)

    # Line chart
    with right, st.container(border=True):
        line_yaxis = st.selectbox("Feature over time", possible_features)
        line_df = filtered_df[['day_of_year', line_yaxis]].groupby('day_of_year').mean().reset_index()
        linefig = px.line(line_df, x='day_of_year', y=line_yaxis,title=f"Average {line_yaxis.replace('_', ' ')} by Day of Year",labels={'day_of_year': 'Day of year'})
        st.plotly_chart(linefig, use_container_width=True)

    # Scatter
    with st.container(border=True):
        sx, sy = st.columns(2)
        xscatter = sx.selectbox("X axis", possible_features, index=0)
        yscatter = sy.selectbox("Y axis", possible_features, index=1)
        scatterfig = px.scatter(filtered_df, x=xscatter, y=yscatter, color='day_type',color_discrete_map=DAY_COLORS,title=f"{yscatter} vs {xscatter}",labels={'day_type': 'Day type'})
        scatterfig.update_layout(title_x=0.5)
        st.plotly_chart(scatterfig, use_container_width=True)