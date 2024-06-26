# Recommender System for News

By utilising the methods above, we were able to implement a working recommendation system for recommending news articles, satisfying constraints of many small projects which were according to our research not addressed all in any of the state-of-the art work. 
We used a news article in a non-common Czech language. Although our data are from the same source and all the articles are news articles with similar structures, they are cross domain in the sense that they contain a variety of topics and themes. Used methods rely either on no  training, no prior knowledge about the users, or solely on unsupervised training methods since labelling data costs resources. Additionally, we used existing supporting datasets for the hyperparameter tuning or CF-based recommendations, however, no existing or manually collected dataset of user ratings was used and the collected set of articles is unique, so we still emphasise the general re-usability of this approach for different textual data. By promoting our system architecture, we were able to run the system in a cost-effective manner and perform the final user testing on system updating based on new data (new published articles) and new user actions.
We found an interesting link between the positive assessment of the recommendations by the users and the model picked based on analogies and similarities tests, which may provide a way to limit the number of models used in the production for user testing.
Text-similarity-based methods were successful, and positively rated by the group of users, in providing recommendations of news articles. The most successful recommendation approach showed 76% average precision. The characteristics of this approach were:
●	The use of the Word2Vec    model.
●	The Word2Vec model was trained based on the parameters assessed by the word similarities tests based on the WordSim353 dataset where the similarities judgements of a model are compared with the human judged similarities.
●	Particularly, the best-performing model was selected based on the out-of-vocabulary ratio representing the ratio of pairs with words unknown to the model, as a leading metric along with the prerequisite condition of having a significant correlation between the human assessed similarities and the model calculated similarities.
●	Recommendations based on finding document similarities of one document to other documents, computed by the Cosine similarity distances of the word embedding vectors derived by a Word2Vec model.
●	We managed to achieve better results when we trained the models on the same textual data as the final recommended items, as opposed to using large Wikipedia textual data.
●	We emphasized the approach of not using any prior knowledge or dataset about the user ratings and general usability of the recommendation system for a project of a smaller scale with limited resources.
The trade-off between the analogy tests and the non-vocabulary ratio, as well as the highest performance in the analogy test, did not provide an improvement over the default variant of the Word2Vec parameters. On the contrary, the Word2Vec model with the lowest loss of vocabulary (out-of-vocabulary ratio) representing the loss of words during the training performed significantly better than the Word2Vec, and Doc2Vec variants trained on the default hyperparameters, and in the final task of providing similar articles also outperformed the TF-IDF method. 
This CB method along with the articles vectorized by the TF-IDF method, was superficial to the CF and (fuzzy) hybrid approaches inspired by some movie recommendation systems, e.g., the methods of the recommendation system Predictory (Walek & Fojtik, 2020). Although predicting the user ratings with the SVD method used on the user-item matrix is a well-established method, a cold start problem with a few user evaluations was too significant. Trying to address this problem by using generated rankings based on the article popularity Reuters data did not help.
TF-IDF method provides a great opportunity for implementing simple text-based recommendations effectively without prolonged time of development. Word2Vec model trained on hyperparameters selected by relatively simple presented method with the cosine distance measure provides opportunity to effectively to further enhance the recommendation results and provide a greater “intelligence” behind the recommended results.
The presented architecture, methods and technologies were shown to be appropriate for building a recommender system of a smaller scale in restricted conditions.
