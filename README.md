# Disaster Response Pipeline Project


<kbd> <img src="https://github.com/ChristopherCochet/Disaster-Message-NLP-Pipeline/blob/master/images/project-overview.PNG"/> </kbd>


**Project description:** In this project we explore the processing and classification of disaster event related tweets so that rescue and support organizations can quickly identify which tweets might require their immediate attention and follow-up 'on-theground' action.  It includes training and deploying a machine learning pipeline that categorizes these events messages so that these messages can be appropriatelty routed to an disaster relief agency.<br>

> 'We have a lot of problem at Delma 75 Avenue Albert Jode, those people need water and food.' <br><br>


<kbd> <img src="https://github.com/ChristopherCochet/Disaster-Message-NLP-Pipeline/blob/master/images/App-demo.gif"/> </kbd>


## 1. Disaster Response Dataset

 The data set contains 26,248 real messages that were sent during disaster events. These messages are classified as either 'direct', 'social' or 'news' and those are the labesl we will use to train the nlp model.

<kbd> <img src="https://github.com/ChristopherCochet/Disaster-Message-NLP-Pipeline/blob/master/images/dataset-1.PNG"/> </kbd>

 If we look at the dataset once tranformed using the dimension reduction technique latent semantic analysis (LSA) we see the following: 

<kbd> <img src="https://github.com/ChristopherCochet/Disaster-Message-NLP-Pipeline/blob/master/images/dataset-lsa.PNG"/> </kbd>

## 2. ETL Pipeline

<kbd> <img src="https://christophercochet.github.io/Market-Basket-Analysis/images/jupyter.png"/> </kbd>
Refer to the following ETL notebook [here](https://github.com/ChristopherCochet/Disaster-Message-NLP-Pipeline/blob/master/notebooks/ETL%20Pipeline%20Preparation.ipynb)

In the script directory, the 'process_data.py' file perfoms the data cleaning pipeline that:

* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

- To run ETL pipeline that cleans data and stores in database <br>
    ```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```

## 3. Machine Learning Text Classifier Pipeline

<kbd> <img src="https://christophercochet.github.io/Market-Basket-Analysis/images/jupyter.png"/> </kbd>
Refer to the following ML classifier training and tuning notebook [here](http://localhost:8888/notebooks/Disaster-Recovery-Message_Classification/notebooks/ML%20Pipeline%20Preparation.ipynb)

In the script directory, the 'train_classifier.py' file performs the machine learning pipeline that:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using a SearchCV approach
* Outputs results on the test set
* Exports the final model as a pickle file

<kbd> <img src="https://github.com/ChristopherCochet/Disaster-Message-NLP-Pipeline/blob/master/images/model-pipelines.PNG"/> </kbd>

- To run ML pipeline that trains classifier and saves <br>
    ```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```

## 4. Flask Web App
We repurpose a web app where an emergency worker can input a new message and ten get classification results in several categories. The app also outputs a visual of how the machine learning model made the classification. 
In addition to flask, the web app template that also uses html, css, javascript and Plotly visualizations. The model's classification decision rely on the LIME library.

> LIME (local interpretable model-agnostic explanations) is a package for explaining the predictions made by machine learning algorithms. 
> Lime supports explanations for individual predictions from a wide range of classifiers, and support for scikit-learn is built in.

reference :
* https://lime-ml.readthedocs.io/en/latest/ <br>
* https://christophm.github.io/interpretable-ml-book/lime.html <br>

To start the web app locally, run the following command in the app directory and then go to http://localhost:3001 to interact with the app:
    ```python run.py```

<kbd> <img src="https://github.com/ChristopherCochet/Disaster-Message-NLP-Pipeline/blob/master/images/classification-result.PNG"/> </kbd>


## Resources Used for this project
* Udacity Data Science Nanodegree: [here](https://www.udacity.com/course/data-scientist-nanodegree--nd025) <br>
* Build a LIME explainer dashboard with the fewest lines of code: [here](https://towardsdatascience.com/build-a-lime-explainer-dashboard-with-the-fewest-lines-of-code-bfe12e4592d4) <br>
* How to solve 90% of NLP problems: a step-by-step guide: [here](https://blog.insightdatascience.com/how-to-solve-90-of-nlp-problems-a-step-by-step-guide-fda605278e4e) <br>






