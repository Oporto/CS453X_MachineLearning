import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def fPC(y, yhat):
    compared = np.equal(y, yhat)
    return compared.shape[0] / y.shape[0]


def predict_per_predictor(predictor, image):
    r1, c1, r2, c2 = predictor
    bool = image[r1][c1] > image[r2][c2]
    return 1 if bool else 0


def predict_image(image, predictors):
    vectorized_predictor = np.vectorize(predict_per_predictor)
    predicts = vectorized_predictor(np.array(predictors), image)
    avg_prediction = np.mean(predicts)
    return 1 if avg_prediction > 0.5 else 0


def measureAccuracyOfPredictors(predictor, X, y, previous_predictors):
    if predictor in previous_predictors or predictor[:2] == predictor[-2:]:
        return 0
    vectorized_predict = np.vectorize(predict_image)
    previous_predictors.append(predictor)
    yhat = vectorized_predict(X, previous_predictors)
    return fPC(y, yhat)


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


def stepwiseRegression(trainingFaces, trainingLabels, testingFaces,
                       testingLabels):
    possible_pixels = np.array(np.meshgrid(np.arange(24),
                                           np.arang(24))).T.reshape(-1, 2)
    possible_features = np.array(
        np.meshgrid(possible_pixels, possible_pixels)).T.reshape(-1, 4)
    del possible_pixels

    best_predictors = []
    vectorized_accuracy = np.vectorize(measureAccuracyOfPredictors)

    for _ in range(5):
        predictor_results = vectorized_accuracy(
            possible_features, trainingFaces, trainingLabels, best_predictors)
        best_feature = possible_features[np.argmax(predictor_results)]
        best_predictors.append(best_feature)
    
    return best_predictors


def loadData(which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels


if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    training_results = stepwiseRegression(trainingFaces[:10], trainingLabels[:10])
    print(training_results)