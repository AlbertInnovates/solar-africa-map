import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import google.generativeai as genai
import os

# Page configuration
st.set_page_config(
    page_title="Solar Map of Africa",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("solar_data.csv")
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure solar_data.csv exists.")
        return pd.DataFrame()

df = load_data()

# Sidebar
st.sidebar.title("🌍 Solar Map Settings")

# API Key handling
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)

st.sidebar.markdown("---")
st.sidebar.header("Select Location")

selected_city_name = st.sidebar.selectbox(
    "Choose a city to analyze:",
    options=df["City"].tolist() if not df.empty else []
)

# Filter data for selected city
selected_city_data = df[df["City"] == selected_city_name].iloc[0] if not df.empty else None

# Main content
st.title("☀️ Solar Map of Africa")
st.markdown("Explore solar irradiance data and investment potential for major African cities.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Interactive Map")
    if not df.empty:
        # Create map centered on Africa
        m = folium.Map(location=[0, 20], zoom_start=3, tiles="cartodbdark_matter")

        # Add markers
        for index, row in df.iterrows():
            color = "orange" if row["City"] == selected_city_name else "blue"
            tooltip = f"{row['City']}, {row['Country']} - GHI: {row['GHI']} kWh/m²/day"

            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=8,
                popup=tooltip,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(m)

        st_folium(m, width="100%", height=500)
    else:
        st.warning("No data available to display on map.")

with col2:
    st.subheader("City Data")
    if selected_city_data is not None:
        st.metric(label="City", value=f"{selected_city_data['City']}, {selected_city_data['Country']}")
        st.metric(label="Global Horizontal Irradiance (GHI)", value=f"{selected_city_data['GHI']} kWh/m²/day")

        st.markdown("### Location Details")
        st.write(f"**Latitude:** {selected_city_data['Latitude']}")
        st.write(f"**Longitude:** {selected_city_data['Longitude']}")
    else:
        st.info("Select a city to view details.")

# AI Analysis Section
st.markdown("---")
st.subheader("🤖 AI Investment Analysis")

if selected_city_data is not None:
    if st.button("Generate Investment Report"):
        if not api_key:
            st.error("Please provide a Gemini API Key in the sidebar to use this feature.")
        else:
            with st.spinner("Asking Gemini about investment potential..."):
                try:
                    model = genai.GenerativeModel('gemini-pro')
                    prompt = f"""
                    Act as a solar energy investment expert.
                    Analyze the solar investment potential for {selected_city_data['City']}, {selected_city_data['Country']}.
                    The Global Horizontal Irradiance (GHI) is {selected_city_data['GHI']} kWh/m²/day.

                    Please provide a brief report covering:
                    1. Suitability for solar power based on the GHI.
                    2. General economic factors relevant to solar investment in this region (mentioning 'Investment Potential').
                    3. Potential risks and opportunities.

                    Keep the tone professional and concise.
                    """
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"An error occurred while communicating with Gemini: {e}")

# Footer
st.markdown("---")
st.caption("Solar Map of Africa - Demo Application")
