import json
import plotly
import pandas as pd
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA, TruncatedSVD

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objects import Scatter
from sqlalchemy import create_engine
from lime import lime_text
from lime.lime_text import LimeTextExplainer 


model_file_name = "classifier.pkl"
model_path = "models/"+model_file_name

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def generate_LSA(p_data, p_labels, p_model):
    # Build a Dataframe used to identify missclassified Genres, add the LSA (2 components score)
    data_transform = model[:-1].transform(p_data)
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(data_transform)
    lsa_scores = lsa.transform(data_transform)
    frame = { 'Message': p_data, 'Lsa Score 1': lsa_scores[:,0], 'Lsa Score 2': lsa_scores[:,1], 'Labels' : p_labels } 
    return(pd.DataFrame(frame))

# load data
print("loading messages from database ...")
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)


# load model
print("loading model {} ...".format(model_path))
model = pickle.load(open(model_path, 'rb'))


#data to plot
print("generating scatter plot dataframe ...")
df_lsa = generate_LSA(df['message'], df['genre'], model)
df_lsa['Message'] = df_lsa['Message'].str.replace('.','<br>')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    
    # extract data needed for visuals
    df_lsa_direct = df_lsa[df_lsa['Labels'] == 'direct']
    df_lsa_social = df_lsa[df_lsa['Labels'] == 'social']
    df_lsa_news = df_lsa[df_lsa['Labels'] == 'news']
    
    # create visuals
    graphs = [
        {
            'data': [
                Scatter(
                    x=df_lsa_direct['Lsa Score 1'],
                    y=df_lsa_direct['Lsa Score 2'],
                    mode = 'markers',
                    name='direct messages',
                    text='Genre:' + df_lsa_direct['Labels'] + '<br>' + 'Message:' + df_lsa_direct['Message'],
                    marker= dict( 
                                    symbol = 'triangle-left',
                                    color=df_lsa_direct['Labels'].map({'direct': 'green', 'news': 'blue', 'social': 'red'})
                                ) 
                ),
                Scatter(
                    x=df_lsa_social['Lsa Score 1'],
                    y=df_lsa_social['Lsa Score 2'],
                    mode = 'markers',
                    name='social messages',
                    text='Genre:' + df_lsa_social['Labels'] + '<br>' + 'Message:' + df_lsa_social['Message'],
                    marker= dict( 
                                    symbol = 'triangle-left',
                                    color=df_lsa_social['Labels'].map({'direct': 'green', 'news': 'blue', 'social': 'red'})
                                ) 
                ),
                Scatter(
                    x=df_lsa_news['Lsa Score 1'],
                    y=df_lsa_news['Lsa Score 2'],
                    mode = 'markers',
                    name='news messages',
                    text='Genre:' + df_lsa_news['Labels'] + '<br>' + 'Message:' + df_lsa_news['Message'],
                    marker= dict( 
                                    symbol = 'triangle-left',
                                    color=df_lsa_news['Labels'].map({'direct': 'green', 'news': 'blue', 'social': 'red'})
                                ) 
                )                                
            ],

            'layout': {
                'showlegend': True,
                'title': 'Latent Semantic Analysis (LSA - 2 components) of the Messages by Genres',
                'width': 1200,
                'height': 1000,
                'yaxis': {
                    'title': "LSA Component 1"
                },
                'xaxis': {
                    'title': "LSA Component 2"
                },
            }
        }
    ]

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    #query = request.form['query']

    # use model to predict classification for query
    print("generating classification prediction for message {}...".format(query))
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # set-up Lime
    classes = df.columns[4:].to_list()
    print("classes = {}".format(classes))
    limeexplainer = LimeTextExplainer(class_names = classes)
    exp = limeexplainer.explain_instance(query, model.predict_proba, num_features = 10, top_labels=3)     

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        exp=exp.as_html(),
        model = model[-1],
        classification_result=classification_results
    )



def main():
    app.run(host='localhost', port=3001, debug=True)


if __name__ == '__main__':
    main()