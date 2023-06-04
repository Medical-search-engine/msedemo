
import streamlit as st  #importing streamlit library

import pandas as pd

import numpy as np
import gensim
from gensim.models import Word2Vec
#from gensim.models import FastText
from matplotlib import pyplot


import matplotlib.pyplot as plt
import plotly.graph_objects as go     # our main display package
import string # used for preprocessing
import re # used for preprocessing
import nltk # the Natural Language Toolkit, used for preprocessing
import numpy as np # used for managing NaNs
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords # used for preprocessing
from nltk.stem import WordNetLemmatizer # used for preprocessing

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

import pymongo
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(**st.secrets["mongo"])

client=init_connection()
@st.cache_data(ttl=600)
def get_data():
    db=client.mydb
    items=db.mycollection.find()
    items=list(items)

items=get_data()
for item in items:
    st.write(f"{item['name']}")

df=pd.read_csv('input/Dimension-covid.csv')   #for preprocessing
df1 = pd.read_csv('input/Dimension-covid.csv')  # for returning results

# function to remove all urls
def remove_urls(text):    
    new_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
    return new_text

# make all text lowercase
def text_lowercase(text):
    return text.lower()

# remove numbers
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result

# remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# tokenize
def tokenize(text):
    text = word_tokenize(text)
    return text

# remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    text = [i for i in text if not i in stop_words]
    return text

# lemmatize Words 
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    text = [lemmatizer.lemmatize(token) for token in text]
    return text

#Creating one function so that all functions can be applied at once
def preprocessing(text):
    
    text = text_lowercase(text)
    text = remove_urls(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    text = ' '.join(text)
    return text


skipgram = Word2Vec.load('output/model_Skipgram.bin')
#FastText=Word2Vec.load('output/model_Fasttext.bin')



# In[12]:


vector_size=100   #defining vector size for each word


# Calculate the mean Vector 
def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in tokenize(words) if word in list(word2vec_model.wv.index_to_key)]
    if len(words) >= 1:
        return np.mean(word2vec_model.wv[words], axis=0)
    else:
        return np.array([0]*100)



# Read the Skipgram embedding for abstract 
K=pd.read_csv('Data/skipgram-vec.csv')   

K2=[]                          
for i in range(df.shape[0]):
    K2.append(K[str(i)].values)


# Read the FastText embedding for the abstract 
KK=pd.read_csv('Data/FastText-vec.csv')

K1=[]
for i in range(df.shape[0]):
    K1.append(KK[str(i)].values)



from numpy import dot
from numpy.linalg import norm

# Cosing similary Functions
def cos_sim(a,b):

    return dot(a, b)/(norm(a)*norm(b)) 


pd.set_option("display.max_colwidth", 0)       #this function will display full text from each column




#streamlit function 
def main():
    # Load data and models
    data = df1     #our data which we have to display
    st.title("Medical Search engine")      #title of our app
    st.write('Select Model')       #text below title

    
    Vectors = st.selectbox("Model",options=['Skipgram' ])
    if Vectors=='Skipgram':
        K=K2
        word2vec_model=skipgram
    #elif Vectors=='Fasttext':
     #   K=K1
        #word2vec_model=FastText

    st.write('Type your query here')

    query = st.text_input("Search box")   #getting input from user

    def preprocessing_input(query):
            
            query=preprocessing(query)
            query=query.replace('\n',' ')
            K=get_mean_vector(word2vec_model,query)
   
        
            return K   
    # Get the top 10 results using cosine similarity
    def top_n(query,p,df1):
        
        
        query=preprocessing_input(query)   
                                    
        x=[]
    
        for i in range(len(p)):
            
            x.append(cos_sim(query,p[i]))
        tmp=list(x)    
        res = sorted(range(len(x)), key = lambda sub: x[sub])[-10:]
        sim=[tmp[i] for i in reversed(res)]
        print(sim)

        L=[]
        for i in reversed(res):
           
    
            L.append(i)
        return df1.iloc[L, [1,2,5,6]],sim  
    
    model = top_n
    if query:
        
        P,sim =model(str(query),K,data)     #storing our output dataframe in P
        #Plotly function to display our dataframe in form of plotly table
        fig = go.Figure(data=[go.Table(header=dict(values=['ID', 'Title','Abstract','Publication Date','Score']),cells=dict(values=[list(P['Trial ID'].values),list(P['Title'].values), list(P['Abstract'].values),list(P['Publication date'].values),list(np.around(sim,4))],align=['center','right']))])
        #displying our plotly table
        fig.update_layout(height=1700,width=700,margin=dict(l=0, r=10, t=20, b=20))
        
        st.plotly_chart(fig) 
        # Get individual results
        
####import streamlit as st

def intro():
    import streamlit as st

    st.write("# Welcome to Medical Search Engine! ")
    st.sidebar.success("Select a Module.")

    st.markdown(
        """
        Searching for medical information on the Web is a challenging task for ordinary Internet users. Often, users are uncertain about their exact medical situations, are unfamiliar with medical terminology, and hence have difficulty in coming up with the right search keywords.[2] An intelligent medical search engine is specifically designed to address this challenge. It uses several techniques to improve its usability and search result quality. First, it uses an interactive questionnaire-based query interface to guide users to provide the most important information about their situations. Users perform search by selecting symptoms and answering questions rather than by typing keyword queries. Second, it uses medical knowledge (e.g., diagnostic decision trees) to automatically form multiple queries from a user' answers to the questions. These queries are used to perform search simultaneously. Third, it provides various kinds of help functions.
    """
    )

def mapping_demo():
    import streamlit as st
    import pandas as pd
    import pydeck as pdk

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write(
        """
        This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/library/api-reference/charts/st.pydeck_chart)
to display geospatial data.
"""
    )

    @st.cache_data
    def from_data_file(filename):
        url = (
            "http://raw.githubusercontent.com/streamlit/"
            "example-data/master/hello/v1/%s" % filename
        )
        return pd.read_json(url)

    try:
        ALL_LAYERS = {
            "Bike Rentals": pdk.Layer(
                "HexagonLayer",
                data=from_data_file("bike_rental_stats.json"),
                get_position=["lon", "lat"],
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                extruded=True,
            ),
            "Bart Stop Exits": pdk.Layer(
                "ScatterplotLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_color=[200, 30, 0, 160],
                get_radius="[exits]",
                radius_scale=0.05,
            ),
            "Bart Stop Names": pdk.Layer(
                "TextLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_text="name",
                get_color=[0, 0, 0, 200],
                get_size=15,
                get_alignment_baseline="'bottom'",
            ),
            "Outbound Flow": pdk.Layer(
                "ArcLayer",
                data=from_data_file("bart_path_stats.json"),
                get_source_position=["lon", "lat"],
                get_target_position=["lon2", "lat2"],
                get_source_color=[200, 30, 0, 160],
                get_target_color=[200, 30, 0, 160],
                auto_highlight=True,
                width_scale=0.0001,
                get_width="outbound",
                width_min_pixels=3,
                width_max_pixels=30,
            ),
        }
        st.sidebar.markdown("### Map Layers")
        selected_layers = [
            layer
            for layer_name, layer in ALL_LAYERS.items()
            if st.sidebar.checkbox(layer_name, True)
        ]
        if selected_layers:
            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state={
                        "latitude": 37.76,
                        "longitude": -122.4,
                        "zoom": 11,
                        "pitch": 50,
                    },
                    layers=selected_layers,
                )
            )
        else:
            st.error("Please choose at least one layer above.")
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )

def plotting_demo():
    import streamlit as st
    import time
    import numpy as np

    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    chart = st.line_chart(last_rows)

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text("%i%% Complete" % i)
        chart.add_rows(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)

    progress_bar.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")


def data_frame_demo():
    import streamlit as st
    import pandas as pd
    import altair as alt

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
    

    @st.cache_data
    def get_UN_data():
        AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
        df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
        return df.set_index("Region")

    try:
        df = get_UN_data()
        countries = st.multiselect(
            "Choose countries", list(df.index), ["China", "United States of America"]
        )
        if not countries:
            st.error("Please select at least one country.")
        else:
            data = df.loc[countries]
            data /= 1000000.0
            st.write('Query metrics', data.sort_index())

            data = data.T.reset_index()
            data = pd.melt(data, id_vars=["index"]).rename(
                columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
            )
            chart = (
                alt.Chart(data)
                .mark_area(opacity=0.3)
                .encode(
                    x="year:T",
                    y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                    color="Region:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )

page_names_to_funcs = {
    "—": intro,
    "Graphical Representation": plotting_demo,
    "Geological Illustration": mapping_demo,
    "DataFrame Visualisation": data_frame_demo
}

demo_name = st.sidebar.selectbox("Choose a Module", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()



if __name__ == "__main__":
    main()

