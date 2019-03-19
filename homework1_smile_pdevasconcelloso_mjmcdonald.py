import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt


class Predictor(object):
    def __init__(self, r1, c1, r2, c2):
        self.r1 = r1
        self.r2 = r2
        self.c1 = c1
        self.c2 = c2

    def tuple(self):
        return self.r1, self.c1, self.r2, self.c2

    def __repr__(self):
        return self.tuple().__str__()

def fPC(y, yhat):
    misses = np.nonzero(y - yhat)
    return 1 - (misses[0].shape[0] / y.shape[0])


def predict_per_predictor(predictor, image):
    r1, c1, r2, c2 = predictor
    bool = image[r1][c1] > image[r2][c2]
    return 1 if bool else 0


def predict_image(image, predictors):
    image = image.reshape(6, 6)
    # vectorized_predictor = np.vectorize(predict_per_predictor)
    # predicts = vectorized_predictor(np.array(predictors), image)
    predicts = np.apply_along_axis(lambda pred: predict_per_predictor(pred, image), 1, np.array(predictors))
    avg_prediction = np.mean(predicts)
    return 1 if avg_prediction > 0.5 else 0


def measureAccuracyOfPredictors(predictor, X, y, previous_predictors):
    predictor = list(predictor)
    if predictor in previous_predictors or (predictor[0] == predictor[2] and predictor[1] == predictor[3]):
        return float(0)
    # vectorized_predict = np.vectorize(predict_image)
    # vectorized_predict.excluded.add(1)
    predictors = previous_predictors.copy()
    predictors.append(predictor)
    
    yhat = np.apply_along_axis(lambda img: predict_image(img, predictors), 1, X)
    # yhat = vectorized_predict(X, previous_predictors)
    acc = fPC(y, yhat)
    return acc


def display_predictors(im, predictors):
    # Show an arbitrary test image in grayscale
    fig, ax = plt.subplots(1)
    ax.imshow(im, cmap='gray')
    for r1, c1, r2, c2 in predictors:
        # Show r1,c1
        rect = patches.Rectangle(
            (c1, r1), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle(
            (c2, r2), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    # Display the merged result
    plt.show()


def stepwiseRegression(trainingFaces, trainingLabels):
    possible_features = np.array(
        np.meshgrid(np.arange(6), np.arange(6), np.arange(6), np.arange(6))).T.reshape(-1, 4)
    # possible_features = np.array(list(map(lambda x: Predictor(x[0],x[1],x[2],x[3]), possible_features)))

    best_predictors = []
    # vectorized_accuracy = np.vectorize(measureAccuracyOfPredictors)
    # vectorized_accuracy.excluded.add(1)
    # vectorized_accuracy.excluded.add(2)
    # vectorized_accuracy.excluded.add(3)

    for j in range(5):
        predictor_results = np.apply_along_axis(
            lambda pred: measureAccuracyOfPredictors(pred, trainingFaces, trainingLabels, best_predictors),
            1, possible_features)
        # predictor_results = vectorized_accuracy(possible_features, trainingFaces, trainingLabels, best_predictors)
        
        best_feature = possible_features[np.argmax(predictor_results)]
        best_predictors.append(list(best_feature))
        print("added", best_feature, "after round", j, "with acc ", predictor_results[np.argmax(predictor_results)])
    
    return best_predictors


def loadData(which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 6*6)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels


if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    training_results = stepwiseRegression(trainingFaces[:100], trainingLabels[:100])
    print(training_results)