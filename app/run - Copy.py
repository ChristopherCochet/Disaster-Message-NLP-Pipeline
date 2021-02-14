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
app = Flask(__name__)

def generate_LSA(data, labels, model):
    # Build a Dataframe used to identify missclassified Genres, add the LSA (2 components score)
    data_transform = model[:-1].fit_transform(data)
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(data_transform)
    lsa_scores = lsa.transform(data_transform)
    frame = { 'Message': data, 'Lsa Score 1': lsa_scores[:,0], 'Lsa Score 2': lsa_scores[:,1], 'Labels' : labels } 
    return(pd.DataFrame(frame))

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load model
model = pickle.load(open("models/"+model_file_name, 'rb'))

mystr = "#earthquake"
res = model.predict([mystr])[0]
print("############# {}".format(res))

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

#data to plot
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
@app.route('/go', methods=['POST'])
def go():
    # save user input in query
    #query = request.args.get('query', '') 
    query = request.form['query']
    print("*********** {}".format(model))
    print("------ {}".format(mystr))

    # use model to predict classification for query
    classification_labels = model.predict([mystr])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # set-up Lime
    limeexplainer = LimeTextExplainer(class_names = model.classes_)
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