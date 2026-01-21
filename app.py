import streamlit as st
import folium
from streamlit_folium import st_folium  
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import pycountry


country_code_map = {
    country.name: country.alpha_2.lower()
    for country in pycountry.countries
}
def get_flag_url(country_name):
    code = country_code_map.get(country_name)
    if code:
        return f"https://flagcdn.com/w80/{code}.png"
    return None
from predictor import predict_olympic_medals, predict_olympic_medals_detailed, predict_event_podium
df = pd.read_csv("features.csv")
year_data = pd.read_csv("data_clean.csv")
results = pd.read_csv("olympic_results.csv")
country_list = df["country_name"].unique().tolist()

st.set_page_config(page_title="Olympic Medal Prediction App", page_icon="🏅")
st.title("Olympic Medal Prediction App 🏅")


tab1, tab2, tab3 = st.tabs(["Overall Prediction", "Event Podium Prediction", "Country Medal History"])



with tab1:
    country_list = df["country_name"].unique().tolist()
    st.title("Detailed Medal Breakdown") 
    st.write("See predicted counts of individual, doubles, and team medals for the next Olympic games.")
    selected_country = st.selectbox("Select a country:", country_list, key = "detailed_country")
    if selected_country:
        flag_url = get_flag_url(selected_country)
        if flag_url:
            st.image(flag_url, width=60)
    if selected_country:
        try:
            predictions = predict_olympic_medals_detailed(selected_country)
    
            st.subheader(f"Detailed Medal Breakdown for {predictions['country']}")

            st.markdown("### ☀️Summer Olympics")
            st.write(f"🏄 **Individual Medals:** {predictions['summer']['individual_medals']}")
            st.write(f"🧑‍🤝‍🧑 **Doubles Medals:** {predictions['summer']['doubles_medals']}")
            st.write(f"🏅 **Team Medals:** {predictions['summer']['team_medals']}")
            st.write(f"🏆 **Total Summer Medals:** {predictions['summer']['total_medals']}")

            st.markdown("### ❄️Winter Olympics")
            st.write(f"⛷️ **Individual Medals:** {predictions['winter']['individual_medals']}")
            st.write(f"🧑‍🤝‍🧑 **Doubles Medals:** {predictions['winter']['doubles_medals']}")
            st.write(f"🏅 **Team Medals:** {predictions['winter']['team_medals']}")
            st.write(f"🏆 **Total Winter Medals:** {predictions['winter']['total_medals']}")

        except ValueError as e:
            st.error(str(e))
            
       
with tab2:
    st.title("Podium Prediction by Event")
    st.write("Select a sport discipline and event to see the predicted podium finish for next olympics.")

    all_disciplines = sorted(results['discipline_title'].dropna().unique())
    discipline = st.selectbox("Select a discipline:", all_disciplines)

    if discipline:
        event_subset = results[results['discipline_title'] == discipline]['event_title'].dropna().unique()
        event_title = st.selectbox("Select an event:", sorted(event_subset))

        if event_title:
            try:
                podium = predict_event_podium(discipline, event_title)
                st.subheader(f"Predicted Podium for {discipline} - {event_title}")
                medal_emojis = {"gold": "🥇", "silver": "🥈", "bronze": "🥉"}

                for medal, info in podium['podium'].items():
                    emoji = medal_emojis.get(medal.lower(), "🏅")
                    st.markdown(f"### {emoji} {medal.capitalize()}: {info['country']}")
                    st.write(f"- **Probability:** {info['probability']}")
                    st.write(f"- Historical Medals: {info['historical_golds']} Gold, {info['historical_silvers']} Silver, {info['historical_bronzes']} Bronze")
                    st.write(f"- **Total Appearances:** {info['total_appearances']}")
            except ValueError as e:
                st.error(str(e))

with tab3:
    st.title("Country Medal History")
    st.write("Select a country to see its historical medal counts in the Summer and Winter Olympics.")
    
    selected_country = st.selectbox("Select a country:", country_list, key="country_history")
    if selected_country:
        flag_url = get_flag_url(selected_country)
        if flag_url:
            st.image(flag_url, width=60)
    # Filter data for the selected country
    country_data = year_data[year_data['country_name'] == selected_country].copy()
    
    # Only keep rows with a medal
    medal_data = country_data[country_data['medal_type'].notna()].copy()

    # Count medals by year and discipline
    medal_counts = medal_data.groupby(['year', 'discipline_title']) \
        .size().reset_index(name='total_medals')

    # Plot
    fig = px.bar(medal_counts, x='year', y='total_medals',
                 color='discipline_title',
                 title=f'Olympic Medals by Discipline - {selected_country}',
                 barmode='stack')

    st.plotly_chart(fig)

