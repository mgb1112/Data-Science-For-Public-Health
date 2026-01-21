## Description

This class had two different capstone projects. Their descriptions will be listed below. Both projects were completed with Kate Brown.

## Project One
For our project, we loaded data on cases and deaths from the archived time series of the Center for Systems Science and Engineering COVID Dashboard. We then created a function that aggregated the data by country and transposed the data frame so that the dates became rows instead of columns. This allowed us to obtain a count of cases per date for each country.
Since our Streamlit app included a radio button that allowed users to choose between viewing cumulative or daily case counts, we added an if statement in the function to compute the daily counts when that option was selected. Additionally, we included a dropdown menu containing all the unique countries in the original dataset, enabling users to select any country and view the daily or cumulative number of cases or deaths for that country. When the country is selected, two plots show, one containing the daily or cumulative case counts and the other showing the daily or cumulative deaths for that country. We also created a map that shows the location of the country that the user selected so that they are able to understand where the case and death counts are coming from. 

Data from: https://github.com/CSSEGISandData/COVID-19/tree/master/archived_data/archived_time_series
![capstone_appgif](https://github.com/user-attachments/assets/e623a58c-56d9-4c67-9799-65b46af5c224)

## Project Two 
