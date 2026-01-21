import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

data = pd.read_csv("olympic_results.csv")
data_clean = data.drop(columns = ['participant_type', 'athletes', 'rank_equal', 'country_code', 'athlete_url', 'athlete_full_name', 'value_unit', 'value_type'])
# split 'slug_game' into 'city' and 'year'
data_clean[['city', 'year']] = data['slug_game'].str.rsplit('-', n=1, expand=True)
# drop 'slug_game' column
data_clean = data_clean.drop(columns=['slug_game'])
# convert year to int
data_clean['year'] = data_clean['year'].astype(int)
# define known summer and winter olympics years
summer_years = set(range(1896, 2025, 4)) - {1916, 1940, 1944}  # excluded cancelled years
winter_years = set(range(1924, 2025, 4)) | {1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022} - {1940, 1944}
# assign season
data_clean['season'] = data_clean['year'].apply(
    lambda y: 'summer' if y in summer_years else 'winter' if y in winter_years else 'unknown'
)
# add column for gender (men, women, mixed)
data_clean['gender'] = data_clean['event_title'].str.extract(r'(?i)(men|women|mixed)')[0].str.lower()
# add column for team/doubles/individual event
conditions = [
    data_clean['event_title'].str.contains(r'doubles', case=False, na=False),
    data_clean['event_title'].str.contains(r'team', case=False, na=False)
]
choices = ['doubles', 'team']
data_clean['event_type'] = np.select(conditions, choices, default='individual')

# medal information dataframe
# filter only rows where a medal was awarded
medals_only = data_clean[data_clean['medal_type'].notna()].copy()
# group and aggregate counts
medals_df = medals_only.groupby('country_name').agg(
    total_medals=('medal_type', 'count'),
    total_gold_medals=('medal_type', lambda x: (x.str.lower() == 'gold').sum()),
    total_silver_medals=('medal_type', lambda x: (x.str.lower() == 'silver').sum()),
    total_bronze_medals=('medal_type', lambda x: (x.str.lower() == 'bronze').sum()),
    total_summer_medals=('season', lambda x: (x == 'summer').sum()),
    total_winter_medals=('season', lambda x: (x == 'winter').sum()),
    total_team_medals=('event_type', lambda x: (x == 'team').sum()),
    total_doubles_medals=('event_type', lambda x: (x == 'doubles').sum()),
    total_individual_medals=('event_type', lambda x: (x == 'individual').sum()),
    total_mens_medals=('gender', lambda x: (x == 'Men').sum()),
    total_womens_medals=('gender', lambda x: (x == 'Women').sum()),
    total_mixed_medals =('gender', lambda x: (x == 'Mixed').sum())
).reset_index()

# additional country features dataframe
url = "https://www.worldometers.info/gdp/gdp-by-country/"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
# find the table with GDP data
table = soup.find('table')  
# convert the table to a DataFrame
gdp_df = pd.read_html(str(table))[0]
# clean the dataframe by dropping unwanted columns
gdp_df = gdp_df.iloc[:, [1,2,5]]
# rename the columns
gdp_df.columns = ['country_name', 'gdp', 'population']
# drop the $ in the gdp column and convert to numeric
gdp_df = gdp_df.replace({'\$': '', ',': ''}, regex=True)
gdp_df['gdp'] = pd.to_numeric(gdp_df['gdp'], errors='coerce')
print(gdp_df.head())

# features data frame (merge medals and gdp dataframes)
features_df = pd.merge(medals_df, gdp_df, on='country_name', how='left')
print(features_df.head())
# add average summer and winter medals per year
features_df['avg_summer_medals_per_year'] = features_df['total_summer_medals'] / (30)
features_df['avg_winter_medals_per_year'] = features_df['total_winter_medals'] / (23)
# create columns for summer/winter medals by individual/doubles/teams
# group by counts
summer_individual_counts = medals_only[(medals_only['season'] == 'summer') & (medals_only['event_type'] == 'individual')].groupby('country_name')['medal_type'].count().to_dict()
summer_doubles_counts    = medals_only[(medals_only['season'] == 'summer') & (medals_only['event_type'] == 'doubles')].groupby('country_name')['medal_type'].count().to_dict()
summer_team_counts       = medals_only[(medals_only['season'] == 'summer') & (medals_only['event_type'] == 'team')].groupby('country_name')['medal_type'].count().to_dict()

winter_individual_counts = medals_only[(medals_only['season'] == 'winter') & (medals_only['event_type'] == 'individual')].groupby('country_name')['medal_type'].count().to_dict()
winter_doubles_counts    = medals_only[(medals_only['season'] == 'winter') & (medals_only['event_type'] == 'doubles')].groupby('country_name')['medal_type'].count().to_dict()
winter_team_counts       = medals_only[(medals_only['season'] == 'winter') & (medals_only['event_type'] == 'team')].groupby('country_name')['medal_type'].count().to_dict()
# fill in features_df 
features_df['summer_individual_medals'] = features_df['country_name'].map(summer_individual_counts).fillna(0).astype(int)
features_df['summer_doubles_medals']    = features_df['country_name'].map(summer_doubles_counts).fillna(0).astype(int)
features_df['summer_team_medals']       = features_df['country_name'].map(summer_team_counts).fillna(0).astype(int)

features_df['winter_individual_medals'] = features_df['country_name'].map(winter_individual_counts).fillna(0).astype(int)
features_df['winter_doubles_medals']    = features_df['country_name'].map(winter_doubles_counts).fillna(0).astype(int)
features_df['winter_team_medals']       = features_df['country_name'].map(winter_team_counts).fillna(0).astype(int)
# calculate averages per year
features_df['avg_summer_individual_medals_per_year'] = features_df['summer_individual_medals'] / 30
features_df['avg_summer_doubles_medals_per_year']    = features_df['summer_doubles_medals'] / 30
features_df['avg_summer_team_medals_per_year']       = features_df['summer_team_medals'] / 30

features_df['avg_winter_individual_medals_per_year'] = features_df['winter_individual_medals'] / 23
features_df['avg_winter_doubles_medals_per_year']    = features_df['winter_doubles_medals'] / 23
features_df['avg_winter_team_medals_per_year']       = features_df['winter_team_medals'] / 23

# fill missing values (using median)
imputer = SimpleImputer(strategy='median')
features_df[['gdp', 'population']] = imputer.fit_transform(features_df[['gdp', 'population']])

# create additional features
features_df['gdp_per_capita'] = features_df['gdp'] * 1e6 / features_df['population']  # USD per capita
features_df['medals_per_million'] = features_df['total_medals'] / features_df['population']

# prepare features and targets - using average medals per year as targets
X = features_df[['gdp', 'population', 'gdp_per_capita',
                 'total_medals', 'avg_summer_medals_per_year', 'avg_winter_medals_per_year']]
y_summer = features_df['avg_summer_medals_per_year']
y_winter = features_df['avg_winter_medals_per_year']

# preprocessing
numeric_features = X.columns.tolist()
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features)
])

# regression pipelines
summer_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
winter_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# fit regressors
summer_lr.fit(X, y_summer)
winter_lr.fit(X, y_winter)

# classification targets - using top 10% of average medals per year
top10_summer = y_summer.quantile(0.9)
top10_winter = y_winter.quantile(0.9)
y_summer_class = (y_summer > top10_summer).astype(int)
y_winter_class = (y_winter > top10_winter).astype(int)

# classification pipelines
summer_logreg = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
winter_logreg = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# fit classifiers
summer_logreg.fit(X, y_summer_class)
winter_logreg.fit(X, y_winter_class)

# evaluate models
X_train, X_test, y_train, y_test = train_test_split(X, y_summer, test_size=0.2, random_state=42)
summer_pred = summer_lr.predict(X_test)
print(f"Summer MAE: {mean_absolute_error(y_test, summer_pred):.2f} medals per year")

# classification evaluation
_, X_test_class, _, y_test_class = train_test_split(X, y_summer_class, test_size=0.2, random_state=42)
y_pred_class = summer_logreg.predict(X_test_class)
print(f"Summer Classification Accuracy: {accuracy_score(y_test_class, y_pred_class):.2%}")

# prediction function - now predicting next Olympics medals
def predict_olympic_medals(country_name):
    # check if the country exists in the 'country_name' column
    if country_name not in features_df['country_name'].values:
        raise ValueError(f"Country '{country_name}' not found in features_df.")

    # filter the dataframe to get the information for the given country
    country_data = features_df[features_df['country_name'] == country_name].iloc[0]

    # prepare input data for prediction
    input_data = pd.DataFrame([{
        'gdp': country_data['gdp'],
        'population': country_data['population'],
        'gdp_per_capita': country_data['gdp_per_capita'],
        'total_medals': country_data['total_medals'],
        'avg_summer_medals_per_year': country_data['avg_summer_medals_per_year'],
        'avg_winter_medals_per_year': country_data['avg_winter_medals_per_year']
    }])

    # predict next summer and winter olympic medals (using avg per year as proxy)
    summer_medals = round(summer_lr.predict(input_data)[0])
    winter_medals = round(winter_lr.predict(input_data)[0])

    # predict probabilities for being in the top 10% for summer and winter
    summer_prob = summer_logreg.predict_proba(input_data)[0][1]
    winter_prob = winter_logreg.predict_proba(input_data)[0][1]

    # return the predictions and probabilities
    return {
        'country': country_name,
        'next_summer_medals_pred': max(0, summer_medals),  # ensure non-negative
        'next_winter_medals_pred': max(0, winter_medals),  # ensure non-negative
        'summer_top10_prob': f"{summer_prob:.1%}",
        'winter_top10_prob': f"{winter_prob:.1%}"
    }

# predict the number of individual/team/doubles medals for the next Olympics
# prepare feature matrix
X = features_df[[
    'gdp', 'population', 'gdp_per_capita',
    'avg_summer_individual_medals_per_year', 'avg_summer_doubles_medals_per_year', 'avg_summer_team_medals_per_year',
    'avg_winter_individual_medals_per_year', 'avg_winter_doubles_medals_per_year', 'avg_winter_team_medals_per_year'
]]
# initialize regression models
models = {}
for season in ['summer', 'winter']:
    for medal_type in ['individual', 'doubles', 'team']:
        model_name = f'{season}_{medal_type}'
        target = f'avg_{season}_{medal_type}_medals_per_year'
        models[model_name] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        models[model_name].fit(X, features_df[target])
# prediction function (medal counts only)
def predict_olympic_medals_detailed(country_name):
    if country_name not in features_df['country_name'].values:
        raise ValueError(f"Country '{country_name}' not found in dataset.")

    country_data = features_df.loc[features_df['country_name'] == country_name].copy().iloc[0]

    input_data = pd.DataFrame([{
        'gdp': country_data['gdp'],
        'population': country_data['population'],
        'gdp_per_capita': country_data['gdp_per_capita'],
        'avg_summer_individual_medals_per_year': country_data['avg_summer_individual_medals_per_year'],
        'avg_summer_doubles_medals_per_year': country_data['avg_summer_doubles_medals_per_year'],
        'avg_summer_team_medals_per_year': country_data['avg_summer_team_medals_per_year'],
        'avg_winter_individual_medals_per_year': country_data['avg_winter_individual_medals_per_year'],
        'avg_winter_doubles_medals_per_year': country_data['avg_winter_doubles_medals_per_year'],
        'avg_winter_team_medals_per_year': country_data['avg_winter_team_medals_per_year']
    }])

    results = {'country': country_name, 'summer': {}, 'winter': {}}

    for season in ['summer', 'winter']:
        season_total = 0
        for medal_type in ['individual', 'doubles', 'team']:
            model_name = f'{season}_{medal_type}'
            pred = models[model_name].predict(input_data)[0]
            pred = max(0, round(pred))  # Ensure non-negative integer predictions
            results[season][f'{medal_type}_medals'] = pred
            season_total += pred

        results[season]['total_medals'] = season_total

    return results

# podium prediction by event
def predict_event_podium(discipline, event_title, season=None, top_n=3):

    # determine season if not provided
    if season is None:
        winter_disciplines = [
            'Alpine Skiing', 'Biathlon', 'Bobsleigh', 'Cross-Country Skiing', 
            'Curling', 'Figure Skating', 'Freestyle Skiing', 'Ice Hockey', 
            'Luge', 'Nordic Combined', 'Short Track Speed Skating', 
            'Skeleton', 'Ski Jumping', 'Snowboard', 'Speed Skating'
        ]
        season = 'winter' if discipline in winter_disciplines else 'summer'
    
    # filter data for the chosen event
    event_data = data_clean[
        (data_clean['discipline_title'].str.lower() == discipline.lower()) &
        (data_clean['event_title'].str.lower().str.contains(event_title.lower())) & 
        (data_clean['season'] == season)
    ].copy()
    
    if event_data.empty:
        raise ValueError(f"No historical data found for {discipline} - {event_title} in {season} Olympics")
    
    # prep features and targets
    features = []
    targets = []
    country_stats = []
    
    # obtain participating countries
    participating_countries = event_data['country_name'].unique()
    
    for country in participating_countries:
        country_data = event_data[event_data['country_name'] == country]
        
        # features
        total_appearances = len(country_data)
        gold_count = len(country_data[country_data['medal_type'].str.lower() == 'gold'])
        silver_count = len(country_data[country_data['medal_type'].str.lower() == 'silver'])
        bronze_count = len(country_data[country_data['medal_type'].str.lower() == 'bronze'])
        medal_rate = (gold_count + silver_count + bronze_count) / total_appearances if total_appearances > 0 else 0
        
        # get country's features from features_df
        country_features = features_df[features_df['country_name'] == country]
        if not country_features.empty:
            gdp = country_features['gdp'].values[0]
            population = country_features['population'].values[0]
            gdp_per_capita = country_features['gdp_per_capita'].values[0]
            total_medals = country_features['total_medals'].values[0]
        else:
            # defaults for missing data
            gdp = features_df['gdp'].median()
            population = features_df['population'].median()
            gdp_per_capita = features_df['gdp_per_capita'].median()
            total_medals = 0
        
        features.append([
            total_appearances,
            gold_count,
            silver_count,
            bronze_count,
            medal_rate,
            gdp,
            population,
            gdp_per_capita,
            total_medals
        ])
        
        # create target for regression (weighted medal score)
        targets.append(gold_count*3 + silver_count*2 + bronze_count*1)
        
        country_stats.append({
            'country': country,
            'appearances': total_appearances,
            'golds': gold_count,
            'silvers': silver_count,
            'bronzes': bronze_count,
            'medal_score': gold_count*3 + silver_count*2 + bronze_count*1
        })
    
    # Fallback to simple historical counts if insufficient data
    if len(features) < 3:
        print("Warning: Insufficient data for ML models - falling back to historical medal counts")
        return create_podium_from_history(country_stats, discipline, event_title, season, top_n)
    
    # convert to numpy arrays
    X = np.array(features)
    y_reg = np.array(targets)
    
    # normalize targets for classification
    try:
        # try to create classes - at least 25% must have medals to attempt classification
        if sum(y_reg > 0) >= max(3, len(y_reg)*0.25):
            y_class = (y_reg >= np.percentile(y_reg[y_reg > 0], 25)).astype(int) 
            can_use_classifier = len(np.unique(y_class)) > 1
        else:
            can_use_classifier = False
    except:
        can_use_classifier = False
    
    # train regression model
    reg_model = make_pipeline(StandardScaler(), LinearRegression())
    reg_model.fit(X, y_reg)
    
    # train classifier [if we have multiple classes]
    if can_use_classifier:
        class_model = make_pipeline(StandardScaler(), LogisticRegression())
        class_model.fit(X, y_class)
    
    # predict for all countries
    predictions = []
    for i, country in enumerate(participating_countries):
        x = X[i].reshape(1, -1)
        
        # regression prediction (medal score)
        medal_score = max(0, reg_model.predict(x)[0])
        
        # classification prediction if available, otherwise use normalized medal score
        if can_use_classifier:
            podium_prob = class_model.predict_proba(x)[0][1]
        else:
            # fallback: use softmax of medal scores
            podium_prob = np.exp(medal_score) / np.sum(np.exp([s['medal_score'] for s in country_stats]))
        
        predictions.append({
            'country': country,
            'medal_score': medal_score,
            'podium_probability': podium_prob,
            **country_stats[i]
        })
    
    # sort by medal score and get top_n countries
    predictions.sort(key=lambda x: x['medal_score'], reverse=True)
    top_predictions = predictions[:top_n]
    
    # normalize probabilities to sum to 1 among top_n
    total_prob = sum(p['podium_probability'] for p in top_predictions)
    if total_prob > 0:
        for pred in top_predictions:
            pred['podium_probability'] /= total_prob
    
    return create_podium_structure(top_predictions, discipline, event_title, season, top_n, can_use_classifier)

def create_podium_from_history(country_stats, discipline, event_title, season, top_n):
    """Fallback function when ML models can't be used"""
    # sort by historical medal score
    country_stats.sort(key=lambda x: x['medal_score'], reverse=True)
    top_countries = country_stats[:top_n]
    
    # create probabilities based on historical medal share
    total_score = sum(c['medal_score'] for c in top_countries)
    if total_score == 0:
        # equal probability if no medals won
        for c in top_countries:
            c['podium_probability'] = 1.0 / len(top_countries)
    else:
        for c in top_countries:
            c['podium_probability'] = c['medal_score'] / total_score
    
    return create_podium_structure(top_countries, discipline, event_title, season, top_n, False)

def create_podium_structure(predictions, discipline, event_title, season, top_n, used_ml):
    """Creates the final output structure"""
    podium = {}
    medals = ['gold', 'silver', 'bronze'][:top_n]
    
    for i, pred in enumerate(predictions[:top_n]):
        medal = medals[i] if i < len(medals) else f'position_{i+1}'
        
        podium[medal] = {
            'country': pred['country'],
            'probability': f"{pred['podium_probability']:.1%}",
            'medal_score': pred['medal_score'],
            'historical_golds': pred['golds'],
            'historical_silvers': pred['silvers'],
            'historical_bronzes': pred['bronzes'],
            'total_appearances': pred['appearances']
        }
    
    return {
        'discipline': discipline,
        'event': event_title,
        'season': season,
        'podium': podium,
        }

data_clean['medal_awarded'] = data_clean['medal_type'].notna().astype(int)

medal_counts = data_clean[data_clean['medal_type'].notna()] \
    .groupby(['country_name', 'year'])['medal_type'] \
    .count().reset_index().rename(columns={'medal_type': 'total_medals'})

if 'total_medals' in data_clean.columns:
    data_clean = data_clean.drop(columns='total_medals')

data_clean = data_clean.merge(medal_counts, on=['country_name', 'year'], how='left')

