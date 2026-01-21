import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk

# Load data
confirmed_cases = pd.read_csv(
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
)
deaths = pd.read_csv(
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
)


def get_country_data(df, country, case_type):
    # Aggregate by country
    df_country = df[df["Country/Region"] == country].drop(columns=["Province/State", "Lat", "Long"])
    df_country = df_country.groupby("Country/Region").sum()

    # Transpose to get dates as rows
    df_country = df_country.T
    df_country.index = pd.to_datetime(df_country.index)
    df_country.columns = ["Count"]

    # Compute daily cases if selected
    if case_type == "Daily":
        df_country["Count"] = df_country["Count"].diff().fillna(0).astype(int)

    return df_country.reset_index().rename(columns={"index": "Date"})


# Streamlit UI
st.title("COVID-19 Information Viewer")

# Dropdown for country selection
countries = confirmed_cases["Country/Region"].unique()
selected_country = st.selectbox("Select Country", sorted(countries))

# Radio button for case type
case_type = st.radio("Select Case Type", ["Cumulative", "Daily"])

# Get data
confirmed_country_data = get_country_data(confirmed_cases, selected_country, case_type)
deaths_country_data = get_country_data(deaths, selected_country, case_type)

st.header(f"Location of {selected_country}")

# Extract latitude, longitude, and latest case count
df_map = confirmed_cases[confirmed_cases["Country/Region"] == selected_country].copy()
df_map = df_map[["Lat", "Long", df_map.columns[-1]]]  # Get the latest column (most recent case count)
df_map.columns = ["latitude", "longitude", "Cases"]

# Remove locations with zero cases
df_map = df_map[df_map["Cases"] > 0]

# Define the initial view with lower zoom
if not df_map.empty:
    view_state = pdk.ViewState(
        latitude=df_map["latitude"].mean(),  # Centering map
        longitude=df_map["longitude"].mean(),
        zoom=2,  # Lower zoom for a wider view
        pitch=0,
    )

    # Define map layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        df_map,
        get_position=["longitude", "latitude"],
        get_radius=150000,  # Adjust dot size
        get_fill_color=[200, 30, 0, 160],  # Red color
        pickable=True,
    )

    # Render the map
    st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=view_state, layers=[layer]))

# Visualize confirmed cases
st.header("COVID-19 Cases Over Time")
fig, ax = plt.subplots()
ax.plot(confirmed_country_data["Date"], confirmed_country_data["Count"], label="Cases", color="blue")
ax.set_xlabel("Date")
ax.set_ylabel("Count")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)
# Visualize deaths
st.header("COVID-19 Deaths Over Time")
fig, ax = plt.subplots()
ax.plot(deaths_country_data["Date"], deaths_country_data["Count"], label="Deaths", color="red")
ax.set_xlabel("Date")
ax.set_ylabel("Count")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)
