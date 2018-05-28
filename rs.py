# Recommender systems
# Find similar users, find movies, recommend movies, test the recommendation acurracy
# Lucas cordeiro da Silva
# UFF - Universidade Federal Fluminense

import matplotlib.pyplot as plt
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


# Prototipo com funcoes para execucao dos metodos Kmeans e KNN
class RecommenderSystem(object):

    kmeans = None
    knn = None

    # Matriz numpy para treino
    X = None

    def __init__(self, matrix):
        self.X = matrix

    # Encontrar o id dos usuários que pertencem ao mesmo cluster
    def ClusterIndicesNumpy(self, clustNum, labels_array):  # numpy
        return np.where(labels_array == clustNum)[0]

    # Encontrar os filmes que o usuário ativo já assistiu
    def getWatchedMovies(self, user_vector):  # numpy
        evaluations = {}
        # pega filmes assistidos pelo usuário ativo
        for i, col in enumerate(user_vector):
            if col > 0:
                evaluations.setdefault(i, None)
                evaluations[i] = col
        return evaluations

    # Montar matriz usuariossimilares-filmes
    def mountSimilarUsersMatrix(self, matrix, users):
        users_vector = []

        for user in users:
            users_vector.append(matrix[user])
        return users_vector

    # Gerar recomendacao de notas para filmes
    def recommendMovies(self, similarusers_matrix, activeuser_whatchedmovies):

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
    def PredictionAccuracy(self, y_true, y_pred):
        tau = stats.kendalltau(y_true, y_pred)[0]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = sqrt(mean_squared_error(y_true, y_pred))

        return {"Tau": tau, "MAE": mae, "RMSE": rmse}

    # Train kmeans
    def trainkmeans(self, X, k):
        # Instanciar o objeto kmeans
        self.kmeans = KMeans(n_clusters=k)

        # Computar k-means clustering
        self.kmeans.fit(X)

    # Train kmeans
    def trainknn(self, X, k):
        # Instanciar o objeto KNN
        self.nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')

        # Computar kNN
        self.nbrs.fit(X)

    # Pegar usuários similares usando K-Means
    def getSimilarUserskmeans(self, matrix, activeuser_vector):

        # Rótulo do cluster que cada usuário pertence
        users_clusterlabel = self.kmeans.labels_

        # Preveja o cluster mais próximo em que cada amostra pertence
        activeuser_clusterlabel = self.kmeans.predict([activeuser_vector])

        # Encontrar todos os usuários que pertencem ao mesmo cluster
        similarusers = self.ClusterIndicesNumpy(
            activeuser_clusterlabel, users_clusterlabel)

        print("kmeans: ", similarusers)

        # Gerar matrix usuariossimilares-filmes
        similarusers_matrix = self.mountSimilarUsersMatrix(
            matrix, similarusers)

        return similarusers_matrix

    # Pegar usuários similares usando KNN
    def getSimilarUsersknn(self, matrix, activeuser_vector):

        # Preveja quais sao os K usuarios mais proximos ao usuario ativo
        distances, similarusers = self.nbrs.kneighbors([activeuser_vector])

        print("knn: ", similarusers)

        # Gerar matrix usuariossimilares-filmes
        similarusers_matrix = self.mountSimilarUsersMatrix(
            matrix, similarusers[0])

        return similarusers_matrix

    # Exibe os gráficos
    def renderCharts(self, results, kvalues):
        # Tau
        kmeans_points_tau = [_["kmeans"]["tau"] for _ in results]
        knn_points_tau = [_["knn"]["tau"] for _ in results]
        plt.figure(1)
        plt.plot(kvalues, kmeans_points_tau, 'o-')
        plt.plot(kvalues, knn_points_tau, 'o-')
        plt.title('K-Means(Blue) vs Knn(Orange)')
        plt.ylabel('Tau')
        plt.xlabel('K')
        plt.xticks(kvalues)
        plt.yticks(kmeans_points_tau + knn_points_tau)

        # MAE
        kmeans_points_mae = [_["kmeans"]["mae"] for _ in results]
        knn_points_mae = [_["knn"]["mae"] for _ in results]
        plt.figure(2)
        plt.plot(kvalues, kmeans_points_mae, 'o-')
        plt.plot(kvalues, knn_points_mae, 'o-')
        plt.title('K-Means(Blue) vs Knn(Orange)')
        plt.ylabel('mae')
        plt.xlabel('K')
        plt.xticks(kvalues)
        plt.yticks(kmeans_points_mae + knn_points_mae)

        # RMSE
        kmeans_points_rmse = [_["kmeans"]["rmse"] for _ in results]
        knn_points_rmse = [_["knn"]["rmse"] for _ in results]
        plt.figure(3)
        plt.plot(kvalues, kmeans_points_rmse, 'o-')
        plt.plot(kvalues, knn_points_rmse, 'o-')
        plt.title('K-Means(Blue) vs Knn(Orange)')
        plt.ylabel('rmse')
        plt.xlabel('K')
        plt.xticks(kvalues)
        plt.yticks(kmeans_points_rmse + knn_points_rmse)

        plt.show()


#
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

        self.matrixtrain = self.matrix[:last]
        self.matrixtest = self.matrix[last:]


# Inicia tudo
def main():

    # trainpercentage = 0.8
    # data = Data("100k/u.data", trainpercentage)
    # # matrix numpy
    # X = np.array(data.matrixtrain)
    # k = 3
    # nbrs = NearestNeighbors(n_neighbors=k)

    # nbrs.fit(X)

    # nbrs.fit([[1, 2, 3, 4, 5], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])

    # distances, indices = nbrs.kneighbors([[1, 1, 1, 1, 1]])
    # print(indices[0])
    # sys.exit()

    # K-Means Method -------------------------------------------

    # Parametros de funcionamento do K
    k_min = 3
    k_max = 13
    step = 1

    # Tamanho da matrix de treinamento, ex: 0.5 -> 50% treinamento e 50% teste
    trainpercentage = 0.70

    # Ler dataset
    # The full u data set, 100000 ratings by 943 users on 1682 items.
    # Each user has rated at least 20 movies.  Users and items are
    # numbered consecutively from 1.  The data is randomly
    # ordered. This is a tab separated list of
    # user id | item id | rating | timestamp.
    # The time stamps are unix seconds since 1/1/1970 UTC
    # https://grouplens.org/datasets/movielens/100k/
    data = Data("100k/u.data", trainpercentage)

    # Lista com os resultados finais variando k
    results = []

    # matrix numpy
    X = np.array(data.matrixtrain)

    # Criar objeto prototipo
    RecommenderSystemd = RecommenderSystem(X)

    # para cada valor de K
    kvalues = [k for k in range(k_min, (k_max + step), step)]
    for k in range(k_min, (k_max + step), step):

        # Metricas de avaliacao da recomendacao
        metrics = {"kmeans": {"tau": [], "mae": [], "rmse": []},
                   "knn": {"tau": [], "mae": [], "rmse": []}}

        print("k: ", k)

        # 1- Treinar conjundo de dados
        RecommenderSystemd.trainkmeans(X, k)
        RecommenderSystemd.trainknn(X, k)

        # para cada usuario de teste
        for user in data.matrixtest:

            # 2- Definir vetor do usuário ativo e rank real
            activeuser_vector = user
            activeuser_whatchedmovies = RecommenderSystemd.getWatchedMovies(
                activeuser_vector)

            # 3- Encontrar usuários similares ao usuário ativo
            # K-Means
            similarusers_matrix = RecommenderSystemd.getSimilarUserskmeans(
                data.matrixtrain, activeuser_vector)

            # KNN
            similarusers_matrixknn = RecommenderSystemd.getSimilarUsersknn(
                data.matrixtrain, activeuser_vector)

            # 4- Recomendar filmes dos usuários similares ao usuário ativo
            # K-Means
            recommendation = RecommenderSystemd.recommendMovies(
                similarusers_matrix, activeuser_whatchedmovies)

            # KNN
            recommendationknn = RecommenderSystemd.recommendMovies(
                similarusers_matrixknn, activeuser_whatchedmovies)

            # 5- Calcular Precisao da previsão
            # K-Means
            tau, mae, rmse = RecommenderSystemd.PredictionAccuracy(
                list(activeuser_whatchedmovies.values()), list(recommendation.values())).values()

            # K-NN
            tauknn, maeknn, rmseknn = RecommenderSystemd.PredictionAccuracy(
                list(activeuser_whatchedmovies.values()), list(recommendationknn.values())).values()

            # Guardar os calculos de precisao
            metrics["kmeans"]["tau"].append(tau)
            metrics["kmeans"]["mae"].append(mae)
            metrics["kmeans"]["rmse"].append(rmse)
            metrics["knn"]["tau"].append(tauknn)
            metrics["knn"]["mae"].append(maeknn)
            metrics["knn"]["rmse"].append(rmseknn)
            # print("Predicted Ratings: ", recommendation, "\n")
            # print("Real ratings: ", activeuser_whatchedmovies, "\n")

        # Encontrar a média dos calculos de precisao e guardar
        metrics["kmeans"]["tau"] = sum(
            x for x in metrics["kmeans"]["tau"]) / len(metrics["kmeans"]["tau"])
        metrics["kmeans"]["mae"] = sum(
            x for x in metrics["kmeans"]["mae"]) / len(metrics["kmeans"]["mae"])
        metrics["kmeans"]["rmse"] = sum(x for x in metrics["kmeans"]["rmse"]
                                        ) / len(metrics["kmeans"]["rmse"])
        metrics["knn"]["tau"] = sum(
            x for x in metrics["knn"]["tau"]) / len(metrics["knn"]["tau"])
        metrics["knn"]["mae"] = sum(
            x for x in metrics["knn"]["mae"]) / len(metrics["knn"]["mae"])
        metrics["knn"]["rmse"] = sum(x for x in metrics["knn"]["rmse"]
                                     ) / len(metrics["knn"]["rmse"])

        # Adiciona o valor das metricas no vetor de resultados
        results.append(metrics)

    # Imprimir resultados
    print("Prediction accuracy: ", results)

    RecommenderSystemd.renderCharts(results, kvalues)

    print("kvalues", kvalues)


if __name__ == "__main__":
    main()
