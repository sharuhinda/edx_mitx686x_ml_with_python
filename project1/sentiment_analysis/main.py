import project1 as p1
import utils
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# Data loading. There is no need to edit code in this section.
#-------------------------------------------------------------------------------

train_data = utils.load_data('reviews_train.tsv')
val_data = utils.load_data('reviews_val.tsv')
test_data = utils.load_data('reviews_test.tsv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

dictionary = p1.bag_of_words(train_texts)

train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)

#-------------------------------------------------------------------------------
# Problem 5
#-------------------------------------------------------------------------------

toy_features, toy_labels = toy_data = utils.load_toy_data('toy_data.tsv')

T = 10
L = 0.2

def plot_toy_results(algo_name, thetas):
    print('theta for', algo_name, 'is', ', '.join(map(str,list(thetas[0]))))
    print('theta_0 for', algo_name, 'is', str(thetas[1]))
    utils.plot_toy_data(algo_name, toy_features, toy_labels, thetas)

thetas_perceptron = p1.perceptron(toy_features, toy_labels, T)
thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, T)
thetas_pegasos = p1.pegasos(toy_features, toy_labels, T, L)

plot_toy_results('Perceptron', thetas_perceptron)
plot_toy_results('Average Perceptron', thetas_avg_perceptron)
plot_toy_results('Pegasos', thetas_pegasos)

# added to evaluate convergence of theta
# def plot_thetas(algo_name, thetas):
#     """
#     Plots the toy data in 2D.
#     Arguments:
#     * features - an Nx2 ndarray of features (points)
#     * labels - a length-N vector of +1/-1 labels
#     * thetas - the tuple (theta, theta_0) that is the output of the learning algorithm
#     * algorithm - the string name of the learning algorithm used
#     """
#     # plot the points with labels represented as colors
#     plt.subplots()
#     # colors = ['b' if label == 1 else 'r' for label in labels]
#     # plt.scatter(features[:, 0], features[:, 1], s=40, c=colors)
#     # xmin, xmax = plt.axis()[:2]

#     # plot the decision boundary
#     # theta, theta_0 = thetas
#     # xs = np.linspace(xmin, xmax)
#     # ys = -(theta[0]*xs + theta_0) / (theta[1] + 1e-16)
#     # plt.plot(xs, ys, 'k-')
#     xs = np.arange(1, thetas.shape[0]+1)
#     plt.plot(xs, thetas[:, 0])
#     plt.plot(xs, thetas[:, 1])

#     # show the plot
#     algo_name = ' '.join((word.capitalize() for word in algo_name.split(' ')))
#     plt.suptitle('Thetas dynamic ({})'.format(algo_name))
#     plt.show()

# thetas_perceptron = None
# thetas_avg_perceptron = None
# thetas_pegasos = None
# for T in np.arange(10, 300, 5):
#     thetas_perceptron = p1.perceptron(toy_features, toy_labels, T)[0] if thetas_perceptron is None else np.vstack((thetas_perceptron, p1.perceptron(toy_features, toy_labels, T)[0]))
#     thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, T)[0].reshape(1, -1) if thetas_avg_perceptron is None else np.vstack((thetas_avg_perceptron, p1.average_perceptron(toy_features, toy_labels, T)[0]))
#     thetas_pegasos = p1.pegasos(toy_features, toy_labels, T, L)[0] if thetas_pegasos is None else np.vstack((thetas_pegasos, p1.pegasos(toy_features, toy_labels, T, L)[0]))

# plot_thetas('Perceptron', thetas_perceptron)
# plot_thetas('Avg perceptron', thetas_avg_perceptron)
# plot_thetas('Pegasos', thetas_pegasos)

#-------------------------------------------------------------------------------
# Problem 7
#-------------------------------------------------------------------------------

T = 10
L = 0.01

pct_train_accuracy, pct_val_accuracy = \
   p1.classifier_accuracy(p1.perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))

avg_pct_train_accuracy, avg_pct_val_accuracy = \
   p1.classifier_accuracy(p1.average_perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))

avg_peg_train_accuracy, avg_peg_val_accuracy = \
   p1.classifier_accuracy(p1.pegasos, train_bow_features,val_bow_features,train_labels,val_labels,T=T,L=L)
print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))

#-------------------------------------------------------------------------------
# Problem 8
#-------------------------------------------------------------------------------

data = (train_bow_features, train_labels, val_bow_features, val_labels)

# values of T and lambda to try
Ts = [1, 5, 10, 15, 25, 50]
Ls = [0.001, 0.01, 0.1, 1, 10]

pct_tune_results = utils.tune_perceptron(Ts, *data)
print('perceptron valid:', list(zip(Ts, pct_tune_results[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(pct_tune_results[1]), Ts[np.argmax(pct_tune_results[1])]))

avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
print('avg perceptron valid:', list(zip(Ts, avg_pct_tune_results[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(avg_pct_tune_results[1]), Ts[np.argmax(avg_pct_tune_results[1])]))

# fix values for L and T while tuning Pegasos T and L, respective
fix_L = 0.01
peg_tune_results_T = utils.tune_pegasos_T(fix_L, Ts, *data)
print('Pegasos valid: tune T', list(zip(Ts, peg_tune_results_T[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(peg_tune_results_T[1]), Ts[np.argmax(peg_tune_results_T[1])]))

fix_T = Ts[np.argmax(peg_tune_results_T[1])]
peg_tune_results_L = utils.tune_pegasos_L(fix_T, Ls, *data)
print('Pegasos valid: tune L', list(zip(Ls, peg_tune_results_L[1])))
print('best = {:.4f}, L={:.4f}'.format(np.max(peg_tune_results_L[1]), Ls[np.argmax(peg_tune_results_L[1])]))

utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
utils.plot_tune_results('Pegasos', 'T', Ts, *peg_tune_results_T)
utils.plot_tune_results('Pegasos', 'L', Ls, *peg_tune_results_L)

#-------------------------------------------------------------------------------
# Use the best method (perceptron, average perceptron or Pegasos) along with
# the optimal hyperparameters according to validation accuracies to test
# against the test dataset. The test data has been provided as
# test_bow_features and test_labels.
#-------------------------------------------------------------------------------

T_best = 25
L_best = 0.01
thetas = p1.pegasos(train_bow_features, train_labels, T_best, L_best)
predictions_test = p1.classify(test_bow_features, thetas[0], thetas[1])
acc_test = p1.accuracy(predictions_test, test_labels)
print('Prediction accuracy for test set: {:.4f}'.format(np.round(acc_test, 4)))

#-------------------------------------------------------------------------------
# Assign to best_theta, the weights (and not the bias!) learned by your most
# accurate algorithm with the optimal choice of hyperparameters.
#-------------------------------------------------------------------------------

best_theta = thetas[0]
wordlist   = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
sorted_word_features = utils.most_explanatory_word(best_theta, wordlist)
print("Most Explanatory Word Features")
print(sorted_word_features[:10])
