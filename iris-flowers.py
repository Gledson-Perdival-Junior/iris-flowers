from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a, b)

class ScrappyKNN():
    def fit(self, features_train, labels_train):
        self.features_train = features_train
        self.labels_train = labels_train

    def predict(self, features_test):
        predictions = []
        for item in features_test:
            #determinar qual outro ponto mais se aproxima
            label = self.closest(item)
            predictions.append(label)

        return predictions
    
    def closest(self, item):
        best_dist = euc(item, self.features_train[0])
        best_index = 0
        for i in range(1, len(self.features_train)):
            dist = euc(item, self.features_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.labels_train[best_index]


iris_dataset = datasets.load_iris()

labels = iris_dataset['target'] 
features = iris_dataset['data']

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = .5)

my_classifier = ScrappyKNN()
#my_classifier = KNeighborsClassifier()
my_classifier.fit(features_train, labels_train)

prediction = my_classifier.predict(features_test)

print("reconhecimento com aproximadamente", round(accuracy_score(labels_test, prediction) * 100) , "% de precisão")