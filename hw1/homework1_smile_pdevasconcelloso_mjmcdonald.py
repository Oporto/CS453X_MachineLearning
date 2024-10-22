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
        
class Image(object):
    def __init__(self, grid):
        self.grid = grid

def fPC(y, yhat):
    misses = np.nonzero(y - yhat)
    return 1 - (misses[0].shape[0] / y.shape[0])


def predict_per_predictor(predictor, image):
    r1, c1, r2, c2 = predictor.tuple()
    print(r1, c1, r2, c2)
    print(image)
    bool = image.grid[r1][c1] > image.grid[r2][c2]
    return 1 if bool else 0


def predict_image(image, predictors):

    image = image.reshape(24, 24)
    # vectorized_predictor = np.vectorize(predict_per_predictor)
    # predicts = vectorized_predictor(np.array(predictors), image)
    predicts = np.apply_along_axis(lambda pred: predict_per_predictor(pred, image), 1, np.array(predictors))

    avg_prediction = np.mean(predicts)
    return 1 if avg_prediction > 0.5 else 0


def measureAccuracyOfPredictors(predictor, X, y, previous_predictors):

    predictor = list(predictor)
    print(predictor)
    if predictor in previous_predictors or (predictor[0] == predictor[2] and predictor[1] == predictor[3]):
        return float(0)
    # vectorized_predict = np.vectorize(predict_image)
    # vectorized_predict.excluded.add(1)

    predictors = previous_predictors.copy()
    predictors.append(predictor)
    
    #yhat = np.apply_along_axis(lambda img: predict_image(img, predictors), 1, X)
    yhat = vectorized_predict(X, previous_predictors)
    acc = fPC(y, yhat)

    return float(acc)



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
    face_objs = np.apply_along_axis(
            lambda im: Image(im),
            1, trainingFaces)
    possible_features = np.array(

    np.meshgrid(np.arange(24), np.arange(24), np.arange(24), np.arange(24))).T.reshape(-1, 4)
 # possible_features = np.array(list(map(lambda x: Predictor(x[0],x[1],x[2],x[3]), possible_features)))
    feature_objs = np.apply_along_axis(
            lambda pred: Predictor(pred[0], pred[1], pred[2], pred[3]),
            1, possible_features)
    best_predictors = []
    vectorized_accuracy = np.vectorize(measureAccuracyOfPredictors)
    vectorized_accuracy.excluded.add(1)
    vectorized_accuracy.excluded.add(2)
    vectorized_accuracy.excluded.add(3)

    for j in range(5):

        print(possible_features.dtype)
        predictor_results = np.apply_along_axis(
            lambda pred: measureAccuracyOfPredictors(pred, trainingFaces, trainingLabels, best_predictors),
            1, possible_features)
        print(predictor_results.dtype)
        # predictor_results = vectorized_accuracy(possible_features, trainingFaces, trainingLabels, best_predictors)
        best_feature = possible_features[np.argmax(predictor_results)]
        best_predictors.append(list(best_feature))
        print("added", best_feature, "after round", j)
    
    return best_predictors


def loadData(which):
    faces = np.load("{}ingFaces.npy".format(which))

    faces = faces.reshape(-1, 24*24)  # Reshape from 576 to 24x24

    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels


def tester(trainingFaces, trainingLabels, testingFaces, testingLabels):
    for sample in [400, 800, 1200, 1600, 2000]:
        predictors = stepwiseRegression(trainingFaces[:sample], trainingLabels[:sample])
        trainingAccuracy = measureAccuracyOfPredictors(predictors, trainingFaces[:sample], trainingLabels[:sample])
        testingAccuracy = measureAccuracyOfPredictors(predictors, testingFaces, testingLabels)
        print("When trained with", sample, "images:")
        print("Training Accuracy:", trainingAccuracy)
        print("Testing Accuracy:", testingAccuracy)


if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    tester(trainingFaces, trainingLabels, testingFaces, testingLabels)