import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as datetime
import pydeck as pdk
from statsmodels.tsa.arima_model import ARIMA

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

st.title('COVID-19 : Analysis And Predictions')

st.markdown('''<span h3 style="font-size:20px"><h3>A Simple Web App built Dashboard Based on COVID-19</h3></span>''',unsafe_allow_html=True)
st.markdown('''<span ><hr style="border-top: 2px solid"></span>''', unsafe_allow_html=True)

st.sidebar.title('Category Sector:')
st.sidebar.markdown('Choose To Access Different Components Of This App :')

df_path = 'Datasets/covid_19_data.csv'
df_confirmed_path = 'Datasets/time_series_covid_19_confirmed.csv'
df_deaths_path = 'Datasets/time_series_covid_19_deaths.csv'
df_recovered_path= 'Datasets/time_series_covid_19_recovered.csv'
df_age_path='Datasets/COVID-19_Cases_Summarized_by_Age_Group.csv'
df_sum_age_path='Datasets/Provisional_COVID-19_Death_Counts_by_Sex__Age__and_State.csv'

@st.cache(persist=True,allow_output_mutation=True)
def get_data(path):
    return pd.read_csv(path)

df=get_data(df_path)
df_confirmed=get_data(df_confirmed_path)
df_deaths=get_data(df_deaths_path)
df_recovered=get_data(df_recovered_path)
df_age=get_data(df_age_path)
df_sum_age=get_data(df_sum_age_path)

# df.drop(['SNo', 'Province/State'], inplace=True, axis=1)
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])

df_confirmed = df_confirmed.replace(np.nan, '', regex=True)
df_deaths = df_deaths.replace(np.nan, '', regex=True)
df_recovered = df_recovered.replace(np.nan, '', regex=True)

df_confirmed_mapd = df_confirmed.copy()
df_confirmed_mapd = df_confirmed_mapd.drop(['Province/State', ], axis=1)
# df_confirmed_mapd.columns=[['Country/Region','lat','long']]
# df_confirmed_mapd

df_age['Last Updated at']=pd.to_datetime(df_age['Last Updated at'])
df_age['Specimen Collection Date']=pd.to_datetime(df_age['Specimen Collection Date'])
df_age_clean=df_sum_age.dropna(axis=1)
df_age_clean=df_age_clean.drop('Sex',axis=1)




# Analysis Of Country :

group_country = df.groupby(['Country/Region', 'ObservationDate']).agg(
    {"Confirmed": 'sum', "Recovered": 'sum', "Deaths": 'sum'})
group_country['Active Cases'] = group_country['Confirmed'] - group_country['Recovered'] - group_country['Deaths']
# Datewise Distribution :
datewise_df = df.groupby(["ObservationDate"]).agg({"Confirmed": 'sum', "Recovered": 'sum', "Deaths": 'sum'})



#Country Wise :

countrywise=df[df["ObservationDate"]==df["ObservationDate"].max()].groupby(["Country/Region"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'}).sort_values(["Confirmed"],ascending=False)
countrywise["Mortality"]=(countrywise["Deaths"]/countrywise["Confirmed"])*100
countrywise["Recovery"]=(countrywise["Recovered"]/countrywise["Confirmed"])*100


#Let us get last 24 hours cases by subtracting :
#Last record - last second record:



country_last_24_confirmed=[]
country_last_24_recovered=[]
country_last_24_deaths=[]
for country in countrywise.index:
    country_last_24_confirmed.append((group_country.loc[country].iloc[-1]-group_country.loc[country].iloc[-2])["Confirmed"])
    country_last_24_recovered.append((group_country.loc[country].iloc[-1]-group_country.loc[country].iloc[-2])["Recovered"])
    country_last_24_deaths.append((group_country.loc[country].iloc[-1]-group_country.loc[country].iloc[-2])["Deaths"])


#Let Us create a dataframe for last 24 HOURS data :

Last_24_Hours_country=pd.DataFrame(list(zip(countrywise.index,country_last_24_confirmed,country_last_24_recovered,
                                            country_last_24_deaths)),
                                   columns=["Country Name","Last 24 Hours Confirmed","Last 24 Hours Recovered",
                                            "Last 24 Hours Deaths"])

#Since we have about 190 Countries , Let us plot Top 10 or 15  Major Countries , Worst Affected By this :

Top_15_Confirmed_24hr=Last_24_Hours_country.sort_values(["Last 24 Hours Confirmed"],ascending=False).head(15)
Top_15_Recoverd_24hr=Last_24_Hours_country.sort_values(["Last 24 Hours Recovered"],ascending=False).head(15)
Top_15_Deaths_24hr=Last_24_Hours_country.sort_values(["Last 24 Hours Deaths"],ascending=False).head(15)

country_last_48_confirmed = []
country_last_48_recovered = []
country_last_48_deaths = []
for country in countrywise.index:
    country_last_48_confirmed.append(
        (group_country.loc[country].iloc[-1] - group_country.loc[country].iloc[-3])["Confirmed"])
    country_last_48_recovered.append(
        (group_country.loc[country].iloc[-1] - group_country.loc[country].iloc[-3])["Recovered"])
    country_last_48_deaths.append((group_country.loc[country].iloc[-1] - group_country.loc[country].iloc[-3])["Deaths"])

Last_48_Hours_country = pd.DataFrame(list(zip(countrywise.index, country_last_48_confirmed, country_last_48_recovered,
                                              country_last_48_deaths)),
                                     columns=["Country Name", "Last 48 Hours Confirmed", "Last 48 Hours Recovered",
                                              "Last 48 Hours Deaths"])



Top_15_Confirmed_48hr=Last_48_Hours_country.sort_values(["Last 48 Hours Confirmed"],ascending=False).head(15)
Top_15_Recoverd_48hr=Last_48_Hours_country.sort_values(["Last 48 Hours Recovered"],ascending=False).head(15)
Top_15_Deaths_48hr=Last_48_Hours_country.sort_values(["Last 48 Hours Deaths"],ascending=False).head(15)


df_age_clean = df_age_clean[df_age_clean['Age group'] != 'All Ages']


confirm=df.groupby('ObservationDate').sum()['Confirmed'].reset_index()
recover=df.groupby('ObservationDate').sum()['Recovered'].reset_index()
death=df.groupby('ObservationDate').sum()['Deaths'].reset_index()





sidebar_list=['Intro','About','Visualizations','Predictions']
select=st.sidebar.selectbox(" ",sidebar_list,key=1)

if(select=='Intro'):
    print('\n')

    st.header('Intro About COVID-19 : ')
    print('\n')
    st.markdown('<style>h3{font-weight: bold;}</style>', unsafe_allow_html=True)
    st.markdown('''<span style="font-weight: bold;">This Web App is Purely Designed For Insightful Analysis and Learning Process ,
                  By Using CSV Based Datasets. Using Machine Learning And Deep Learning Algorithms , I Have Tried
                 To Predict And Analyse The Current Scenario Of Novel Corona Virus or Better Known AS COVID-19. 
                </span>  \n''',unsafe_allow_html=True)

    st.markdown('''<span style="font-weight: bold;">Coronavirus disease (COVID-19) is an inflammation disease from a new virus. The disease causes respiratory ailment (like influenza) with manifestations, for example, cold, cough and fever, and in progressively serious cases, the problem in breathing. COVID-2019 has been perceived as a worldwide pandemic and a few examinations are being led utilizing different numerical models to anticipate the likely advancement of this pestilence. These numerical models dependent on different factors and investigations are dependent upon potential inclination. Here, we presented a model that could be useful to predict the spread of COVID-2019</span>  \n''', unsafe_allow_html=True)
    st.markdown("  \n")
    st.markdown("  \n")

    st.markdown('''<span style="font-weight: bold;">In Order To Access Visualizations and predictions , Please click Category Sector ACCESS button on the left 
                side of this App  </span>''',unsafe_allow_html=True)
    st.markdown("  \n")

    st.subheader('Click Here To Check The Dataset used For This Project : ')

    if st.button("Check Dataset"):
        with st.spinner('Please Wait...Dataset Is Loading : '):
            st.dataframe(df)

    st.markdown('''<span ><hr style="border-top: 2px solid"></span>''', unsafe_allow_html=True)
    st.subheader('  \n'
                'I Have Used The Following Libraries For Making This Project :   \n'
                '  \n')

    st.info("1. Basic Python Libraries :  \n"
            "Pandas  \n"
            "Numpy  \n"
            "Matplotlib  \n"
            "Seaborn  \n"
            "  \n"
            "2. External Data Visualization Library :  \n"
            "Altair  \n"
            "Plotly  \n"
            "  \n"
            "3. Other Important Libraries :  \n"
            "Pydeck  \n"
            "ARIMA Model [Forecasting Time Series in Predictions]  \n"
            "  \n")


elif(select=='About'):
    st.header('About')
    st.text('  \n')
    st.markdown('''<span style="font-weight: bold;">This Project Was Created By Harsh Sharma. I Am a Machine Learning and Deep Learning Developer                  And a Data Science Enthusiast.
                You Can Checkout My Other Amazing Projects On My GitHub Profile.  Do Checkout My Linkedin Profile As Well! </span> \n''',unsafe_allow_html=True)
    st.text('  \n')
    st.text('  \n')
    st.text('  \n')

    if st.button("GitHub Profile"):
        st.markdown('[Visit Now!](https://github.com/harshgeek4coder)')
    if st.button("LinkedIn Profile"):
        st.markdown('[Visit Now!](https://www.linkedin.com/in/harsh-sharma-484a4ab6/)')
    if st.button("Portfolio"):
        st.markdown('[Visit Now!](https://theharshsharma.tech/)')



elif(select=='Visualizations'):
    st.header("Visualizations Sector : ")
    viz_list=['Day Wise','Date Wise','Country Wise','Age Wise']
    select_v=st.sidebar.selectbox("Select Time Preference",viz_list,key=2)

    if (select_v=='Day Wise'):

        # Some Preprocessing :








        st.header('Distribution Of Number Of Cases WorldWide : ')
        st.markdown('Please Select The Type Of Case To See Plots And Visualizations : ')

        types_case = ['Confirmed', 'Recovered', 'Active', 'Deaths', 'Closed']

        num_of_per=['Overall','1 Month','2 Months','3 Months','4 Months','5 Months','6 Months','7 Months']

        select1 = st.selectbox('Please Select Type Of Case : ', types_case)

        select2=st.selectbox('Please Select Duration : ',num_of_per)


        if (select1 == 'Confirmed'):

            if(select2=='Overall'):
                st.subheader("Overall RConfirmed Cases")



                # Confirmed Cased Day Wise :

                fig = px.bar(datewise_df,x=datewise_df.index, y=datewise_df["Confirmed"],color='Confirmed')
                fig.update_layout(title={'text': "Distribution of Number of Confirmed Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif(select2=='1 Month'):

                fig = px.bar(datewise_df, x=datewise_df.index[:32], y=datewise_df["Confirmed"][:32], color=datewise_df["Confirmed"][:32])
                fig.update_layout(title={'text': "Distribution of Number of Confirmed Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '2 Months'):

                fig = px.bar(datewise_df, x=datewise_df.index[:62], y=datewise_df["Confirmed"][:62],
                             color=datewise_df["Confirmed"][:62])
                fig.update_layout(title={'text': "Distribution of Number of Confirmed Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '3 Months'):

                fig = px.bar(datewise_df, x=datewise_df.index[:92], y=datewise_df["Confirmed"][:92],
                             color=datewise_df["Confirmed"][:92])
                fig.update_layout(title={'text': "Distribution of Number of Confirmed Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '4 Months'):

                fig = px.bar(datewise_df, x=datewise_df.index[:122], y=datewise_df["Confirmed"][:122],
                             color=datewise_df["Confirmed"][:122])
                fig.update_layout(title={'text': "Distribution of Number of Confirmed Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '5 Months'):

                fig = px.bar(datewise_df, x=datewise_df.index[:152], y=datewise_df["Confirmed"][:152],
                             color=datewise_df["Confirmed"][:152])
                fig.update_layout(title={'text': "Distribution of Number of Confirmed Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '6 Months'):

                fig = px.bar(datewise_df, x=datewise_df.index[:182], y=datewise_df["Confirmed"][:182],
                             color=datewise_df["Confirmed"][:182])
                fig.update_layout(title={'text': "Distribution of Number of Confirmed Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '7 Months'):

                fig = px.bar(datewise_df, x=datewise_df.index[:212], y=datewise_df["Confirmed"][:212],
                             color=datewise_df["Confirmed"][:212])
                fig.update_layout(title={'text': "Distribution of Number of Confirmed Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)





        elif (select1 == 'Recovered'):

            if(select2=='Overall'):
                st.subheader("Overall Recovered Cases")


                # Recovered Cases
                fig = px.bar(x=datewise_df.index, y=datewise_df["Recovered"],color=datewise_df["Recovered"])
                fig.update_layout(title={'text': "Distribution of Number of Recovered Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif(select2=='1 Month'):

                fig = px.bar(x=datewise_df.index[:32], y=datewise_df["Recovered"][:32], color=datewise_df["Recovered"][:32])
                fig.update_layout(title={'text': "Distribution of Number of Recovered Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '2 Months'):

                fig = px.bar(x=datewise_df.index[:62], y=datewise_df["Recovered"][:62],
                             color=datewise_df["Recovered"][:62])
                fig.update_layout(title={'text': "Distribution of Number of Recovered Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '3 Months'):

                fig = px.bar(x=datewise_df.index[:92], y=datewise_df["Recovered"][:92],
                             color=datewise_df["Recovered"][:92])
                fig.update_layout(title={'text': "Distribution of Number of Recovered Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '4 Months'):

                fig = px.bar(x=datewise_df.index[:122], y=datewise_df["Recovered"][:122],
                             color=datewise_df["Recovered"][:122])
                fig.update_layout(title={'text': "Distribution of Number of Recovered Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '5 Months'):

                fig = px.bar(x=datewise_df.index[:152], y=datewise_df["Recovered"][:152],
                             color=datewise_df["Recovered"][:152])
                fig.update_layout(title={'text': "Distribution of Number of Recovered Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '6 Months'):

                fig = px.bar(x=datewise_df.index[:182], y=datewise_df["Recovered"][:182],
                             color=datewise_df["Recovered"][:182])
                fig.update_layout(title={'text': "Distribution of Number of Recovered Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '7 Months'):

                fig = px.bar(x=datewise_df.index[:212], y=datewise_df["Recovered"][:212],
                             color=datewise_df["Recovered"][:212])
                fig.update_layout(title={'text': "Distribution of Number of Recovered Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)


        elif (select1 == 'Active'):

            if(select2=='Overall'):
                st.subheader("Overall Active Cases")

                # Active Cases
                fig = px.bar(x=datewise_df.index, y=datewise_df["Confirmed"] - datewise_df["Recovered"] - datewise_df["Deaths"],
                             color=datewise_df["Confirmed"] - datewise_df["Recovered"] - datewise_df["Deaths"])
                fig.update_layout(title={'text': "Distribution of Number of Active Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '1 Month'):
                # Active Cases
                fig = px.bar(x=datewise_df.index[:32],
                             y=datewise_df["Confirmed"][:32] - datewise_df["Recovered"][:32] - datewise_df["Deaths"][:32],
                             color=datewise_df["Confirmed"][:32] - datewise_df["Recovered"][:32] - datewise_df["Deaths"][:32])
                fig.update_layout(title={'text': "Distribution of Number of Active Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '2 Months'):
                # Active Cases
                fig = px.bar(x=datewise_df.index[:62],
                             y=datewise_df["Confirmed"][:62] - datewise_df["Recovered"][:62] - datewise_df["Deaths"][:62],
                             color=datewise_df["Confirmed"][:62] - datewise_df["Recovered"][:62] - datewise_df["Deaths"][:62])
                fig.update_layout(title={'text': "Distribution of Number of Active Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '3 Months'):
                # Active Cases
                fig = px.bar(x=datewise_df.index[:92],
                             y=datewise_df["Confirmed"][:92] - datewise_df["Recovered"][:92] - datewise_df["Deaths"][:92],
                             color=datewise_df["Confirmed"][:92] - datewise_df["Recovered"][:92] - datewise_df["Deaths"][:92])
                fig.update_layout(title={'text': "Distribution of Number of Active Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '4 Months'):
                # Active Cases
                fig = px.bar(x=datewise_df.index[:122],
                             y=datewise_df["Confirmed"][:122] - datewise_df["Recovered"][:122] - datewise_df["Deaths"][
                                                                                               :122],
                             color=datewise_df["Confirmed"][:122] - datewise_df["Recovered"][:122] - datewise_df[
                                                                                                       "Deaths"][:122])
                fig.update_layout(title={'text': "Distribution of Number of Active Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '5 Months'):
                # Active Cases
                fig = px.bar(x=datewise_df.index[:152],
                             y=datewise_df["Confirmed"][:152] - datewise_df["Recovered"][:152] - datewise_df["Deaths"][
                                                                                               :152],
                             color=datewise_df["Confirmed"][:152] - datewise_df["Recovered"][:152] - datewise_df[
                                                                                                       "Deaths"][:152])
                fig.update_layout(title={'text': "Distribution of Number of Active Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '6 Months'):
                # Active Cases
                fig = px.bar(x=datewise_df.index[:182],
                             y=datewise_df["Confirmed"][:182] - datewise_df["Recovered"][:182] - datewise_df["Deaths"][
                                                                                               :182],
                             color=datewise_df["Confirmed"][:182] - datewise_df["Recovered"][:182] - datewise_df[
                                                                                                       "Deaths"][:182])
                fig.update_layout(title={'text': "Distribution of Number of Active Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '7 Months'):
                # Active Cases
                fig = px.bar(x=datewise_df.index[:212],
                             y=datewise_df["Confirmed"][:212] - datewise_df["Recovered"][:212] - datewise_df["Deaths"][
                                                                                               :212],
                             color=datewise_df["Confirmed"][:212] - datewise_df["Recovered"][:212] - datewise_df[
                                                                                                       "Deaths"][:212])
                fig.update_layout(title={'text': "Distribution of Number of Active Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)



        elif (select1 == 'Deaths'):

            if(select2=='Overall'):
                st.subheader("Overall Death Cases")


            # Deaths Cases:

                fig = px.bar(x=datewise_df.index, y=datewise_df["Deaths"],color=datewise_df["Deaths"])
                fig.update_layout(title={'text': "Distribution of Number of Death Cases",
                                     'y': 0.95,
                                     'x': 0.5,
                                     'xanchor': 'center',
                                     'yanchor': 'top'
                                     },
                              xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                              font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif(select2=='1 Month'):

                fig = px.bar(x=datewise_df.index[:32], y=datewise_df["Deaths"][:32], color=datewise_df["Deaths"][:32])
                fig.update_layout(title={'text': "Distribution of Number of Death Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '2 Months'):

                fig = px.bar(x=datewise_df.index[:62], y=datewise_df["Deaths"][:62], color=datewise_df["Deaths"][:62])
                fig.update_layout(title={'text': "Distribution of Number of Death Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '3 Months'):

                fig = px.bar(x=datewise_df.index[:92], y=datewise_df["Deaths"][:92], color=datewise_df["Deaths"][:92])
                fig.update_layout(title={'text': "Distribution of Number of Death Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '4 Months'):

                fig = px.bar(x=datewise_df.index[:122], y=datewise_df["Deaths"][:122], color=datewise_df["Deaths"][:122])
                fig.update_layout(title={'text': "Distribution of Number of Death Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '5 Months'):

                fig = px.bar(x=datewise_df.index[:152], y=datewise_df["Deaths"][:152], color=datewise_df["Deaths"][:152])
                fig.update_layout(title={'text': "Distribution of Number of Death Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '6 Months'):

                fig = px.bar(x=datewise_df.index[:182], y=datewise_df["Deaths"][:182], color=datewise_df["Deaths"][:182])
                fig.update_layout(title={'text': "Distribution of Number of Death Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '7 Months'):

                fig = px.bar(x=datewise_df.index[:212], y=datewise_df["Deaths"][:212], color=datewise_df["Deaths"][:212])
                fig.update_layout(title={'text': "Distribution of Number of Death Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)


        elif (select1 == 'Closed'):

            if(select2=='Overall'):
                st.subheader("Overall Closed Cases")


                # Closed Cases
                fig = px.bar(x=datewise_df.index, y=datewise_df["Recovered"] + datewise_df["Deaths"]
                             ,color=datewise_df["Recovered"] + datewise_df["Deaths"])
                fig.update_layout(title={'text': "Distribution of Number of Closed Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif(select2=='1 Month'):

                fig = px.bar(x=datewise_df.index[:32], y=datewise_df["Recovered"][:32] + datewise_df["Deaths"][:32]
                             , color=datewise_df["Recovered"][:32] + datewise_df["Deaths"][:32])
                fig.update_layout(title={'text': "Distribution of Number of Closed Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '2 Months'):

                fig = px.bar(x=datewise_df.index[:62], y=datewise_df["Recovered"][:62] + datewise_df["Deaths"][:62]
                             , color=datewise_df["Recovered"][:62] + datewise_df["Deaths"][:62])
                fig.update_layout(title={'text': "Distribution of Number of Closed Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '3 Months'):

                fig = px.bar(x=datewise_df.index[:92], y=datewise_df["Recovered"][:92] + datewise_df["Deaths"][:92]
                             , color=datewise_df["Recovered"][:92] + datewise_df["Deaths"][:92])
                fig.update_layout(title={'text': "Distribution of Number of Closed Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '4 Months'):

                fig = px.bar(x=datewise_df.index[:122], y=datewise_df["Recovered"][:122] + datewise_df["Deaths"][:122]
                             , color=datewise_df["Recovered"][:122] + datewise_df["Deaths"][:122])
                fig.update_layout(title={'text': "Distribution of Number of Closed Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '5 Months'):

                fig = px.bar(x=datewise_df.index[:152], y=datewise_df["Recovered"][:152] + datewise_df["Deaths"][:152]
                             , color=datewise_df["Recovered"][:152] + datewise_df["Deaths"][:152])
                fig.update_layout(title={'text': "Distribution of Number of Closed Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '6 Months'):

                fig = px.bar(x=datewise_df.index[:182], y=datewise_df["Recovered"][:182] + datewise_df["Deaths"][:182]
                             , color=datewise_df["Recovered"][:182] + datewise_df["Deaths"][:182])
                fig.update_layout(title={'text': "Distribution of Number of Closed Cases",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",
                                  font=dict(size=18, color="Black"))
                st.plotly_chart(fig)

            elif (select2 == '7 Months'):

                fig = px.bar(x=datewise_df.index[:212], y=datewise_df["Recovered"][:212] + datewise_df["Deaths"][:212]
                             , color=datewise_df["Recovered"][:212] + datewise_df["Deaths"][:212],width=900,height=600)
                fig.update_layout(title={'text':"Distribution of Number of Closed Cases",
                                         'y':0.95,
                                         'x':0.5,
                                         'xanchor':'center',
                                         'yanchor':'top'
                                         },
                                  xaxis_title="Date", yaxis_title="Number of Cases DAY Wise",font=dict(size=18,color="Black"))
                st.plotly_chart(fig)

    elif(select_v=='Date Wise'):

        dw_val=['Cases','Rates']
        dw_vals=st.sidebar.selectbox("Select Category ",dw_val,key='4')

        if(dw_vals=='Cases'):
            st.subheader("Growth of different types of cases")


            fig = go.Figure()
            fig.add_trace(go.Scatter(x=datewise_df.index, y=datewise_df["Confirmed"],
                                     mode='lines+markers',
                                     name='Confirmed Cases'))
            fig.add_trace(go.Scatter(x=datewise_df.index, y=datewise_df["Recovered"],
                                     mode='lines+markers',
                                     name='Recovered Cases'))
            fig.add_trace(go.Scatter(x=datewise_df.index, y=datewise_df["Deaths"],
                                     mode='lines+markers',
                                     name='Death Cases'))
            fig.update_layout(autosize=False,width=900,height=600,
                              margin=dict(l=0),

                              title={"text":"Growth of different types of cases",
                                     'y':0.95,
                                     'x': 0.5,
                                     'xanchor': 'center',
                                     'yanchor': 'top',
                                     },
                              xaxis_title="Date", yaxis_title="Number of Cases",
                              legend=dict(x=0, y=1, traceorder="normal"),
                              font=dict(size=18,color="Black"))
            st.plotly_chart(fig)

        elif(dw_vals=='Rates'):
            rates_list=['Mortality Rate','Recovery Rate']
            rvals=st.selectbox('Select Rate That You Would Like To See :',rates_list,key=5)

            if(rvals=='Recovery Rate'):
                st.subheader("Overall Recovery Rate")

                fig=px.line(x=datewise_df.index, y=(datewise_df["Recovered"]/datewise_df["Confirmed"])*100)
                fig.update_layout(autosize=False,title="Recovery Rate",margin=dict(l=0,t=35),height=480,width=800, legend=dict(x=-0.1, y=1.2, traceorder="normal"))
                fig.update_xaxes(title_text="Date")
                fig.update_yaxes(title_text="Recovery Rate")
                st.plotly_chart(fig)

            elif(rvals == 'Mortality Rate'):
                st.subheader("Overall Mortality Rate")


                fig=px.line(x=datewise_df.index, y=(datewise_df["Deaths"]/datewise_df["Confirmed"])*100)
                fig.update_layout(autosize=False, title="Mortality Rate", margin=dict(l=0, t=35), height=480, width=800,legend=dict(x=-0.1, y=1.2, traceorder="normal"))

                fig.update_xaxes(title_text="Date")
                fig.update_yaxes(title_text="Mortality Rate")
                st.plotly_chart(fig)



    elif (select_v == 'Country Wise'):
        cw_main_list=['Confirmed','Recovered','Deaths','Active']
        cw_list=['Overall','Last 48 Hours','Last 24 Hours']

        cwp_val=st.sidebar.selectbox("Please Select Period :",cw_list,key=6)



        if(cwp_val=='Overall'):

            cases_n_rates_list=['Cases','Rates']
            cw_tm_list = ['Most Number Of  Cases', 'Least Number Of Cases']
            cases_n_rates_val=st.sidebar.selectbox("Select Type Of Visualization :",cases_n_rates_list,key=9)
            cw_tm_val = st.sidebar.selectbox("Please Select Preference : ", cw_tm_list, key=7)


            if(cases_n_rates_val=='Cases'):

                cw_main_val = st.selectbox("Please Select Type Of Case : ", cw_main_list, key=8)
                if(cw_tm_val=='Most Number Of  Cases'):


                    if(cw_main_val=='Confirmed'):
                        st.subheader("Countries With Overall Confirmed Cases")

                        top_15_confirmed = countrywise.sort_values(["Confirmed"], ascending=False).head(15)
                        fig = px.bar(x=top_15_confirmed["Confirmed"], y=top_15_confirmed.index, orientation='h',
                                     color=top_15_confirmed["Confirmed"])

                        fig.update_layout(yaxis={'categoryorder': 'total ascending'},title={"text":"TOP 15 Countries With Overall Confirmed Cases",
                                             'y':0.95,
                                             'x': 0.5,
                                             'xanchor': 'center',
                                             'yanchor': 'top',
                                             },height=600,width=850,
                                          margin=dict(l=0,r=0,b=0,t=50),font=dict(size=18))
                        fig.update_xaxes(title_text="Number Of Cases")
                        fig.update_yaxes(title_text="Countries")

                        st.plotly_chart(fig)

                    elif (cw_main_val == 'Recovered'):
                        st.subheader("Countries With Overall Recovered  Cases")
                        top_15_recovered = countrywise.sort_values(["Recovered"], ascending=False).head(15)
                        fig = px.bar(x=top_15_recovered["Recovered"], y=top_15_recovered.index, orientation='h',
                                     color=top_15_recovered["Recovered"])

                        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                          title={"text": "TOP 15 Countries With Overall Recovered Cases",
                                                 'y': 0.95,
                                                 'x': 0.5,
                                                 'xanchor': 'center',
                                                 'yanchor': 'top',
                                                 }, height=600, width=850,
                                          margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18))
                        fig.update_xaxes(title_text="Number Of Cases")
                        fig.update_yaxes(title_text="Countries")

                        st.plotly_chart(fig)

                    elif (cw_main_val == 'Deaths'):
                        st.subheader("Countries With Overall Death Cases")

                        top_15_deaths = countrywise.sort_values(["Deaths"], ascending=False).head(15)

                        fig = px.bar(x=top_15_deaths["Deaths"],y=top_15_deaths.index, orientation='h',
                                     color=top_15_deaths["Deaths"])

                        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                          title={"text": "TOP 15 Countries With Overall Death Cases",
                                                 'y': 0.95,
                                                 'x': 0.5,
                                                 'xanchor': 'center',
                                                 'yanchor': 'top',
                                                 }, height=600, width=850,
                                          margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18))
                        fig.update_xaxes(title_text="Number Of Cases")
                        fig.update_yaxes(title_text="Countries")

                        st.plotly_chart(fig)

                    elif (cw_main_val == 'Active'):
                        st.subheader("Countries With Overall Active Cases")

                        countrywise["Active Cases"] = (countrywise["Confirmed"] - countrywise["Recovered"] - countrywise["Deaths"])
                        top_15_active = countrywise.sort_values(["Active Cases"], ascending=False).head(15)

                        fig = px.bar(x=top_15_active["Active Cases"],y=top_15_active.index, orientation='h',
                                     color=top_15_active["Active Cases"])

                        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                          title={"text": "TOP 15 Countries With Overall Active Cases",
                                                 'y': 0.95,
                                                 'x': 0.5,
                                                 'xanchor': 'center',
                                                 'yanchor': 'top',
                                                 }, height=600, width=850,
                                          margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18))
                        fig.update_xaxes(title_text="Number Of Cases")
                        fig.update_yaxes(title_text="Countries")

                        st.plotly_chart(fig)


                elif (cw_tm_val =='Least Number Of Cases'):



                    if (cw_main_val == 'Active'):
                        st.subheader("Countries With Least Number Of Active Cases")

                        countrywise["Active Cases"] = (countrywise["Confirmed"] - countrywise["Recovered"] - countrywise["Deaths"])
                        bottom_15_active=countrywise[countrywise["Active Cases"]>1].sort_values(["Active Cases"],ascending=False).tail(15)


                        fig = px.bar(x=bottom_15_active["Active Cases"],y=bottom_15_active.index, orientation='h',
                                     color=bottom_15_active["Active Cases"])

                        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                          title={"text": "TOP 15 Countries With Least Number Of Active Cases",
                                                 'y': 0.95,
                                                 'x': 0.5,
                                                 'xanchor': 'center',
                                                 'yanchor': 'top',
                                                 }, height=550, width=800,
                                          margin=dict(l=0, r=0, b=0, t=50), font=dict(size=13))
                        fig.update_xaxes(title_text="Number Of Cases")
                        fig.update_yaxes(title_text="Countries")

                        st.plotly_chart(fig)

                    elif (cw_main_val == 'Confirmed'):
                        st.subheader("Countries With Least Number Of Confirmed Cases")

                        Bottom_15_Confirmed_24hr = Last_24_Hours_country[
                            Last_24_Hours_country["Last 24 Hours Confirmed"] > 0].sort_values(
                            ["Last 24 Hours Recovered"], ascending=False).tail(15)

                        fig = px.bar(x=Bottom_15_Confirmed_24hr["Last 24 Hours Confirmed"],y=Bottom_15_Confirmed_24hr["Country Name"], orientation='h',
                                     color=Bottom_15_Confirmed_24hr["Last 24 Hours Confirmed"])

                        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                          title={"text": "TOP 15 Countries With Least Number Of Confirmed Cases",
                                                 'y': 0.95,
                                                 'x': 0.5,
                                                 'xanchor': 'center',
                                                 'yanchor': 'top',
                                                 }, height=500, width=750,
                                          margin=dict(l=0, r=0, b=0, t=50), font=dict(size=14))
                        fig.update_xaxes(title_text="Number Of Cases")
                        fig.update_yaxes(title_text="Countries")

                        st.plotly_chart(fig)


                    elif (cw_main_val == 'Recovered'):
                        st.subheader("Countries With Least Number Of Recoveries")

                        Bottom_15_Recoverd_24hr=Last_24_Hours_country[Last_24_Hours_country["Last 24 Hours Recovered"]>0].sort_values(["Last 24 Hours Recovered"],ascending=False).tail(15)


                        fig = px.bar(x=Bottom_15_Recoverd_24hr["Last 24 Hours Recovered"],y=Bottom_15_Recoverd_24hr["Country Name"], orientation='h',
                                     color=Bottom_15_Recoverd_24hr["Last 24 Hours Recovered"])

                        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                          title={"text": "TOP 15 Countries With Least Number Of Recovered Cases",
                                                 'y': 0.95,
                                                 'x': 0.5,
                                                 'xanchor': 'center',
                                                 'yanchor': 'top',
                                                 }, height=500, width=750,
                                          margin=dict(l=0, r=0, b=0, t=50), font=dict(size=14))
                        fig.update_xaxes(title_text="Number Of Cases")
                        fig.update_yaxes(title_text="Countries")

                        st.plotly_chart(fig)


                    elif (cw_main_val == 'Deaths'):
                        st.subheader("Countries With Least Number Of Deaths")

                        Bottom_15_Deaths_24hr=Last_24_Hours_country[Last_24_Hours_country["Last 24 Hours Deaths"]>0].sort_values(["Last 24 Hours Recovered"],ascending=False).tail(15)


                        fig = px.bar(x=Bottom_15_Deaths_24hr["Last 24 Hours Deaths"],y=Bottom_15_Deaths_24hr["Country Name"], orientation='h',
                                     color=Bottom_15_Deaths_24hr["Last 24 Hours Deaths"])

                        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                          title={"text": "TOP 15 Countries With Least Number Of Deaths Cases",
                                                 'y': 0.95,
                                                 'x': 0.5,
                                                 'xanchor': 'center',
                                                 'yanchor': 'top',
                                                 }, height=500, width=750,
                                          margin=dict(l=0, r=0, b=0, t=50), font=dict(size=14))
                        fig.update_xaxes(title_text="Number Of Cases")
                        fig.update_yaxes(title_text="Countries")

                        st.plotly_chart(fig)











            if (cases_n_rates_val == 'Rates'):
                st.header("Rates Distribution :")
                rates_type_list=['Mortality','Recovery']
                rates_zone_list=['High','Low']
                rates_type_value=st.selectbox("Please Select Type Of Rate",rates_type_list,key=10)
                rates_zone_value=st.selectbox('',rates_zone_list,key=11)

                if(rates_type_value=='Mortality'):

                    if(rates_zone_value=='High'):
                        st.subheader("High Mortality Rate :")

                        countrywise_plot_mortal = countrywise[countrywise["Confirmed"] > 500].sort_values(["Mortality"],ascending=False).head(15)

                        fig = px.bar(x=countrywise_plot_mortal["Mortality"],y=countrywise_plot_mortal.index, orientation='h',
                                     color=countrywise_plot_mortal["Mortality"])

                        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                          title={"text": "TOP 15 Countries With High Mortality Rate",
                                                 'y': 0.95,
                                                 'x': 0.5,
                                                 'xanchor': 'center',
                                                 'yanchor': 'top',
                                                 }, height=600, width=850,
                                          margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18))
                        fig.update_xaxes(title_text="Rate in Percentage")
                        fig.update_yaxes(title_text="Countries")

                        st.plotly_chart(fig)

                    elif(rates_zone_value=='Low'):
                        st.subheader("Low Mortality Rate :")
                        countrywise_plot_mortal = countrywise[countrywise["Confirmed"] > 500].sort_values(["Mortality"],ascending=False).tail(15)

                        fig = px.bar(x=countrywise_plot_mortal["Mortality"],y=countrywise_plot_mortal.index,
                                     orientation='h',
                                     color=countrywise_plot_mortal["Mortality"])

                        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                          title={"text": "TOP 15 Countries With Low Mortality Rate",
                                                 'y': 0.95,
                                                 'x': 0.5,
                                                 'xanchor': 'center',
                                                 'yanchor': 'top',
                                                 }, height=600, width=850,
                                          margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18))
                        fig.update_xaxes(title_text="Rate in Percentage")
                        fig.update_yaxes(title_text="Countries")

                        st.plotly_chart(fig)

                elif (rates_type_value == 'Recovery'):

                    if(rates_zone_value=='High'):
                        st.subheader("High Recovery Rate : ")
                        countrywise_plot_recover = countrywise[countrywise["Confirmed"] > 500].sort_values(["Recovery"],ascending=False).head(15)

                        fig = px.bar(x=countrywise_plot_recover["Recovery"],y=countrywise_plot_recover.index,
                                     orientation='h',
                                     color=countrywise_plot_recover["Recovery"])

                        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                          title={"text": "TOP 15 Countries With High Recovery Rate",
                                                 'y': 0.95,
                                                 'x': 0.5,
                                                 'xanchor': 'center',
                                                 'yanchor': 'top',
                                                 }, height=600, width=850,
                                          margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18))
                        fig.update_xaxes(title_text="Rate in Percentage")
                        fig.update_yaxes(title_text="Countries")

                        st.plotly_chart(fig)

                    elif(rates_zone_value=='Low'):
                        st.subheader("Low Recovery Rate : ")

                        countrywise_plot_recover = countrywise[countrywise["Confirmed"] > 500].sort_values(["Recovery"],ascending=False).tail(15)

                        fig = px.bar(x=countrywise_plot_recover["Recovery"],y=countrywise_plot_recover.index,
                                     orientation='h',
                                     color=countrywise_plot_recover["Recovery"])

                        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                          title={"text": "TOP 15 Countries With Low Recovery Rate",
                                                 'y': 0.95,
                                                 'x': 0.5,
                                                 'xanchor': 'center',
                                                 'yanchor': 'top',
                                                 }, height=600, width=850,
                                          margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18))
                        fig.update_xaxes(title_text="Rate in Percentage")
                        fig.update_yaxes(title_text="Countries")

                        st.plotly_chart(fig)











        elif (cwp_val == 'Last 48 Hours'):
            st.header("Last 48 Hours Cases : ")
            cw_main_val = st.selectbox("Please Select Type Of Case : ", cw_main_list, key=8)

            if (cw_main_val == 'Confirmed'):
                st.subheader("Last 48 Hours Confirmed Cases : ")

                fig = px.bar(x=Top_15_Confirmed_48hr["Last 48 Hours Confirmed"],y=Top_15_Confirmed_48hr["Country Name"],
                orientation='h',color=Top_15_Confirmed_48hr["Last 48 Hours Confirmed"])

                fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                  title={"text": "TOP 15 Countries With Confirmed Cases In Last 48 Hours",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top',
                                         }, height=600, width=850,
                                  margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18))
                fig.update_xaxes(title_text="Number Of Cases")
                fig.update_yaxes(title_text="Countries")

                st.plotly_chart(fig)

            elif (cw_main_val == 'Recovered'):
                st.subheader("Last 48 Hours Recovered Cases : ")

                fig = px.bar(x=Top_15_Recoverd_48hr["Last 48 Hours Recovered"],y=Top_15_Recoverd_48hr["Country Name"],
                             orientation='h', color=Top_15_Recoverd_48hr["Last 48 Hours Recovered"])

                fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                  title={"text": "TOP 15 Countries With Recovered Cases In Last 48 Hours",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top',
                                         }, height=600, width=850,
                                  margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18))
                fig.update_xaxes(title_text="Number Of Cases")
                fig.update_yaxes(title_text="Countries")

                st.plotly_chart(fig)

            elif (cw_main_val == 'Deaths'):
                st.subheader("Last 48 Hours Deaths Cases : ")

                fig = px.bar(x=Top_15_Deaths_48hr["Last 48 Hours Deaths"],y=Top_15_Deaths_48hr["Country Name"],
                             orientation='h', color=Top_15_Deaths_48hr["Last 48 Hours Deaths"])

                fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                  title={"text": "TOP 15 Countries With Death Cases In Last 48 Hours",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top',
                                         }, height=600, width=850,
                                  margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18))
                fig.update_xaxes(title_text="Number Of Cases")
                fig.update_yaxes(title_text="Countries")

                st.plotly_chart(fig)

        elif (cwp_val == 'Last 24 Hours'):
            st.header("Last 24 Hours Cases : ")
            cw_main_val = st.selectbox("Please Select Type Of Case : ", cw_main_list, key=8)

            if (cw_main_val == 'Confirmed'):
                st.subheader("Last 24 Hour DConfirmed Cases")
                fig = px.bar(x=Top_15_Confirmed_24hr["Last 24 Hours Confirmed"],
                             y=Top_15_Confirmed_24hr["Country Name"],
                             orientation='h', color=Top_15_Confirmed_24hr["Last 24 Hours Confirmed"])

                fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                  title={"text": "TOP 15 Countries With Confirmed Cases In Last 24 Hours",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top',
                                         }, height=600, width=850,
                                  margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18))
                fig.update_xaxes(title_text="Number Of Cases")
                fig.update_yaxes(title_text="Countries")

                st.plotly_chart(fig)

            elif (cw_main_val == 'Recovered'):
                st.subheader("Last 24 Hours Recovered Cases")

                fig = px.bar(x=Top_15_Recoverd_24hr["Last 24 Hours Recovered"],y=Top_15_Recoverd_24hr["Country Name"],
                             orientation='h', color=Top_15_Recoverd_24hr["Last 24 Hours Recovered"])

                fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                  title={"text": "TOP 15 Countries With Recovered Cases In Last 24 Hours",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top',
                                         }, height=600, width=850,
                                  margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18))
                fig.update_xaxes(title_text="Number Of Cases")
                fig.update_yaxes(title_text="Countries")

                st.plotly_chart(fig)

            elif (cw_main_val == 'Deaths'):
                st.subheader("Last 24 Hour Deaths")

                fig = px.bar(x=Top_15_Deaths_24hr["Last 24 Hours Deaths"],y=Top_15_Deaths_24hr["Country Name"],
                             orientation='h', color=Top_15_Deaths_24hr["Last 24 Hours Deaths"])

                fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                  title={"text": "TOP 15 Countries With Death Cases In Last 24 Hours",
                                         'y': 0.95,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top',
                                         }, height=600, width=850,
                                  margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18))
                fig.update_xaxes(title_text="Number Of Cases")
                fig.update_yaxes(title_text="Countries")

                st.plotly_chart(fig)

    elif(select_v=='Age Wise'):
        st.header("Age Wise Distributions")
        age_wise_list=['Overall','Age Wise By Country']
        age_wise_val=st.sidebar.selectbox("Select Grouping :",age_wise_list,key=12)

        if(age_wise_val=='Overall'):
            st.subheader('Age Wise Distribution Overall')

            fig = px.bar(y=df_age_clean.index, x=df_age_clean['Age group'])
            fig.update_layout(yaxis_title='Number Of Cumulative Cases',
                              xaxis_title='Age Groups',
                              title={"text": "Distribution Of Cases As Per Age :",
                                     'y': 0.95,
                                     'x': 0.5,
                                     'xanchor': 'center',
                                     'yanchor': 'top',
                                     }, height=600, width=850,
                              margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18)
                              )
            st.plotly_chart(fig)

        elif (age_wise_val == 'Age Wise By Country'):
            st.subheader('Age Wise Distribution By Country')
            countries_list=['United States','Alabama','Alaska','Arizona','California','Colorado','Florida','Georgia',
                            'Hawaii','Illinois','Iowa','Massachusetts','Nevada','New Hampshire','New Jersey','New Mexico'
                            ,'New York City','North Carolina','North Dakota','Ohio','Pennsylvania','South Carolina',
                            'South Dakota','Texas','Virginia','Washington']
            country_age_val = st.selectbox("Select Country", countries_list, key=13)


            def get_graph(c_name):

                countries_list = ['United States', 'Alabama', 'Alaska', 'Arizona', 'California', 'Colorado', 'Florida',
                                  'Georgia',
                                  'Hawaii', 'Illinois', 'Iowa', 'Massachusetts', 'Nevada', 'New Hampshire',
                                  'New Jersey', 'New Mexico'
                    , 'New York City', 'North Carolina', 'North Dakota', 'Ohio', 'Pennsylvania', 'South Carolina',
                                  'South Dakota', 'Texas', 'Virginia', 'Washington']



                for x in countries_list:
                    if (c_name == x):
                        fig = px.bar(x=df_age_clean[df_age_clean['State'] == x]['Age group'],orientation='v',
                                     color=df_age_clean[df_age_clean['State'] == x]['Age group'])


                        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                          title={"text": "Distribution Of Cases Age Wise of {}".format(c_name),
                                                 'y': 0.95,
                                                 'x': 0.5,
                                                 'xanchor': 'center',
                                                 'yanchor': 'top',
                                                 }, height=600, width=850,
                                          margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18))
                        fig.update_xaxes(title_text="Age Groups")
                        fig.update_yaxes(title_text="Number Of Cases")
                return st.plotly_chart(fig)


            get_graph(country_age_val)



elif(select=='Predictions'):
    st.header("Predictions Sector : ")
    st.text('  \n')
    pred_type_list=['Confirmed Cases','Recovered Cases','Death Cases']
    pred_type_val=st.selectbox("Please Select Type Of Case You Want To Predict :",pred_type_list)

    if(pred_type_val=='Confirmed Cases'):
        st.subheader("Confirmed Cases Predictions : ")
        with st.spinner('Confirmed Cases -  Forecasted Values : LOADING... '):
            st.text('  \n')
            
            data = pd.Series(confirm.Confirmed.values, index=confirm.ObservationDate.values)

            model = ARIMA(data, order=(5, 1, 0), missing='drop')
            model_fit = model.fit(disp=0)

            X1 = data.values
            size = int(len(X1) * 0.85)
            train, test = X1[0:size], X1[size:len(X1)]
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model_c = ARIMA(history, order=(5, 1, 0))
                model_c_fit = model_c.fit(disp=0)
                output = model_c_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=np.arange(0, len(train)), y=train, mode='lines', name='Actual'))

            fig.add_trace(go.Scatter(x=np.arange(len(train), len(train) + len(test)),
                                     y=np.array(predictions).reshape(np.array(predictions).shape[0]), mode='lines+markers',
                                     name='Predictions'))


            fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                              title={"text": "Comparison Of Actual VS Predicted Values",
                                     'y': 0.95,
                                     'x': 0.5,
                                     'xanchor': 'center',
                                     'yanchor': 'top',
                                     }, height=450, width=850,
                              margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18))
            fig.update_xaxes(title_text="Number Of Days")
            fig.update_yaxes(title_text="Number Of Cases")
            st.success('Predicted Values :')
            st.plotly_chart(fig)

    elif (pred_type_val == 'Recovered Cases'):
        st.subheader("Recovered Cases Predictions : ")
        with st.spinner('Recovered Cases -  Forecasted Values : LOADING... '):
            st.text('  \n')

            data = pd.Series(recover.Recovered.values, index=recover.ObservationDate.values)

            model = ARIMA(data, order=(5, 1, 0), missing='drop')
            model_fit = model.fit(disp=0)

            X1 = data.values
            size = int(len(X1) * 0.85)
            train, test = X1[0:size], X1[size:len(X1)]
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model_c = ARIMA(history, order=(5, 1, 0))
                model_c_fit = model_c.fit(disp=0)
                output = model_c_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=np.arange(0, len(train)), y=train, mode='lines', name='Actual'))

            fig.add_trace(go.Scatter(x=np.arange(len(train), len(train) + len(test)),
                                     y=np.array(predictions).reshape(np.array(predictions).shape[0]), mode='lines+markers',
                                     name='Predictions'))

            fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                              title={"text": "Recovered Case : Comparison Of Actual VS Predicted Values",
                                     'y': 0.95,
                                     'x': 0.5,
                                     'xanchor': 'center',
                                     'yanchor': 'top',
                                     }, height=450, width=850,
                              margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18))
            fig.update_xaxes(title_text="Number Of Days")
            fig.update_yaxes(title_text="Number Of Cases")
            st.success('Predicted Values :')
            st.plotly_chart(fig)

    elif (pred_type_val == 'Death Cases'):
        st.subheader("Death Cases Predictions : ")
        with st.spinner('Death Cases -  Forecasted Values : LOADING... '):
            st.text('  \n')

            data = pd.Series(death.Deaths.values, index=death.ObservationDate.values)

            model = ARIMA(data, order=(5, 1, 0), missing='drop')
            model_fit = model.fit(disp=0)

            X1 = data.values
            size = int(len(X1) * 0.85)
            train, test = X1[0:size], X1[size:len(X1)]
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model_c = ARIMA(history, order=(5, 1, 0))
                model_c_fit = model_c.fit(disp=0)
                output = model_c_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=np.arange(0, len(train)), y=train, mode='lines', name='Actual'))

            fig.add_trace(go.Scatter(x=np.arange(len(train), len(train) + len(test)),
                                     y=np.array(predictions).reshape(np.array(predictions).shape[0]), mode='lines+markers',
                                     name='Predictions'))

            fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                              title={"text": "Death Case : Comparison Of Actual VS Predicted Values",
                                     'y': 0.95,
                                     'x': 0.5,
                                     'xanchor': 'center',
                                     'yanchor': 'top',
                                     }, height=450, width=850,
                              margin=dict(l=0, r=0, b=0, t=50), font=dict(size=18))
            fig.update_xaxes(title_text="Number Of Days")
            fig.update_yaxes(title_text="Number Of Cases")
            st.success('Predicted Values :')
            st.plotly_chart(fig)




















