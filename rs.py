# Recommender systems
# Find similar users, find movies, recommend movies, test the recommendation acurracy
# Lucas cordeiro da Silva
# UFF - Universidade Federal Fluminense

import numpy as np
import pandas as pd
import operator

# Methods
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import scipy.stats as stats

# math
from math import sqrt


class methodkmeans(object):

    # Encontrar o id dos usuários que pertencem ao mesmo cluster
    @staticmethod
    def ClusterIndicesNumpy(clustNum, labels_array):  # numpy
        return np.where(labels_array == clustNum)[0]

    # Encontrar os filmes que o usuário ativo já assistiu
    @staticmethod
    def getWatchedMovies(user_vector):  # numpy
        evaluations = {}
        # pega filmes assistidos pelo usuário alvo
        for i, col in enumerate(user_vector):
            if col > 0:
                evaluations.setdefault(i, None)
                evaluations[i] = col
        return evaluations

    # Montar matriz usuariossimilares-filmes
    @staticmethod
    def mountSimilarUsersMatrix(matrix, users):
        users_vector = []

        for user in users:
            users_vector.append(matrix[user])
        return users_vector

    # Gerar recomendacao de notas para filmes
    @staticmethod
    def recommendMovies(similarusers_matrix, activeuser_whatchedmovies):

        # Contador de avaliacoes
        ratings = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        recommendation = {}

        # Predizer notas aos filmes assistidos
        for movie in activeuser_whatchedmovies:
            ratings = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for user in similarusers_matrix:
                rating = user[movie]
                if rating > 0:
                    ratings[rating] += 1
            recommendation.setdefault(movie, None)
            # Pega a nota com maior ocorrencia , ex: {1: 3, 2: 7, 3: 19, 4: 26, 5: 6} => 4
            recommendation[movie] = max(
                ratings.items(), key=operator.itemgetter(1))[0]
            # print(ratings, max(
            #     ratings.items(), key=operator.itemgetter(1))[0])

        return recommendation

    # Avaliar Recomendacao usando Kendall Tau, MAE, RMSE
    @staticmethod
    def PredictionAccuracy(y_true, y_pred):
        tau, pvalue = stats.kendalltau(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        return {"Tau": tau, "MAE": mae, "RMSE": rmse}

    # Pegar usuários similares
    @staticmethod
    def getSimilarUsers(matrix, activeuser_vector):

        # matrix numpy
        X = np.array(matrix)

        # Instancear o objeto kmeans
        kmeans = KMeans(n_clusters=4, algorithm="full")

        # Computar k-means clustering
        kmeans.fit(X)

        # Rótulo do cluster que cada usuário pertence
        users_clusterlabel = kmeans.labels_

        # Preveja o cluster mais próximo em que cada amostra pertence
        activeuser_clusterlabel = kmeans.predict([activeuser_vector])

        # print(activeuser_clusterid)
        # print(users_clusterid)

        # Encontrar todos os usuários que pertencem ao mesmo cluster
        similarusers = methodkmeans.ClusterIndicesNumpy(
            activeuser_clusterlabel, users_clusterlabel)

        # print(users)
        # print(matrix[41])

        # Gerar matrix usuariossimilares-filmes
        similarusers_matrix = methodkmeans.mountSimilarUsersMatrix(
            matrix, similarusers)

        # print(users_vector[1])
        return similarusers_matrix


class Data(object):

    # Matrix user-movie
    matrix = []
    matrixtrain = []
    matrixtest = []
    trainpercentage = 0.0

    def __init__(self, ratings_path, trainpercentage):
        self.trainpercentage = trainpercentage
        self.LoadData(ratings_path)

    # Load dataBase
    def LoadData(self, ratings_path):

        # File Read
        colname = ['user', 'movie', 'rating', 'timestamp']
        df = pd.read_table(ratings_path, sep='\t', header=None,
                           names=colname, encoding="ISO-8859-1")

        # Determine the number of unique users and movies
        num_users = len(np.unique(df[['user']].values))
        num_movies = len(np.unique(df[['movie']].values))

        # Matrix of size num_users x num_movies (rows x cols)
        self.matrix = [[0 for _ in range(num_movies)]
                       for _ in range(num_users)]

        # print(num_users, num_movies)

        # Fill in matrix from data
        for index in df.index:
            user = df['user'][index]
            movie = df['movie'][index]
            rating = df['rating'][index]
            self.matrix[user - 1][movie - 1] = rating

        last = int(round(num_users * self.trainpercentage))

        self.matrixtrain = self.matrix[0:last]
        self.matrixtest = self.matrix[last:num_users]


# Inicia tudo
def main():

    # Tamanho da matrix de treinamento, 0.5 => 50% treinamento e 50% teste
    trainpercentage = 0.99
    # Ler dataset
    data = Data("100k/u.data", trainpercentage)

    # K-Means Method -------------------------------------------
    result = {"Tau": [], "MAE": [], "RMSE": []}

    print(len(data.matrixtest))

    for user in data.matrixtest:
        # 1- Definir vetor do usuário ativo e rank real
        activeuser_vector = user
        activeuser_whatchedmovies = methodkmeans.getWatchedMovies(
            activeuser_vector)

        # 2- Encontrar usuários similares ao usuário ativo
        similarusers_matrix = methodkmeans.getSimilarUsers(
            data.matrixtrain, activeuser_vector)

        # 3- Recomendar filmes dos usuários similares ao usuário ativo
        recommendation = methodkmeans.recommendMovies(
            similarusers_matrix, activeuser_whatchedmovies)

        # 4- Precisão da previsão
        Tau, Mae, Rmse = methodkmeans.PredictionAccuracy(
            list(activeuser_whatchedmovies.values()), list(recommendation.values())).values()

        result["Tau"].append(Tau)
        result["MAE"].append(Mae)
        result["RMSE"].append(Rmse)

        print("Predicted Ratings: ", recommendation, "\n")
        print("Real ratings: ", activeuser_whatchedmovies, "\n")

    result["Tau"] = sum(x for x in result["Tau"]) / len(result["Tau"])
    result["MAE"] = sum(x for x in result["MAE"]) / len(result["MAE"])
    result["RMSE"] = sum(x for x in result["RMSE"]) / len(result["RMSE"])

    print("Prediction accuracy: ", result)

    # K-NN Method -------------------------------------------

    # # 1- Definir vetor do usuário ativo e rank real
    # activeuser_vector = data.matrix[942]
    # activeuser_whatchedmovies = methodkmeans.getWatchedMovies(activeuser_vector)

    # # 2- Encontrar usuários similares ao usuário ativo
    # similarusers_matrix = methodkmeans.getSimilarUsers(
    #     data.matrix[0:942], activeuser_vector)

    # # 3- Recomendar filmes dos usuários similares ao usuário ativo
    # movies = methodkmeans.recommendMovies(
    #     similarusers_matrix, activeuser_whatchedmovies)

    # # 4- Prediction accuracy
    # result = methodkmeans.PredictionAccuracy(
    #     list(activeuser_whatchedmovies.values()), list(movies.values()))

    # print("Predicted Ratings: ", movies, "\n")
    # print("Real ratings: ", activeuser_whatchedmovies, "\n")
    # print("Prediction accuracy: ", result)


if __name__ == "__main__":
    main()
