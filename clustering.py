import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster, model_selection, datasets

n_images = 4
n_colors = 3

images = []
X_all, colors_all = [], []

for j in range(n_images):
    img = datasets.make_blobs(n_samples=300, centers=n_colors, cluster_std=1.5)
    images.append(img)
    X, colors = img
    X_all.append(X)
    colors_all.append(colors)

X_all = np.array(X_all)
colors_all = np.array(colors_all)

clfs_dict = dict([
    ('KMeans', cluster.KMeans(n_clusters=n_colors)),
    ('MiniBatchKMeans', cluster.MiniBatchKMeans(n_clusters=n_colors)),
    ('Birch', cluster.Birch(n_clusters=n_colors)),
    ('MeanShift', cluster.MeanShift()),
])

clfs = clfs_dict.items()

scoring = [
    'rand_score',
    'v_measure_score',
    'adjusted_mutual_info_score'
]

fig, ax = plt.subplots(len(clfs) + 1, n_images, figsize=(24, 24))

fig.tight_layout()

for j in range(n_images):
    X = X_all[j]
    colors = colors_all[j]

    x = X[:, 0]
    y = X[:, 1]

    ax[0, j].scatter(x, y, c=colors)
    ax[0, j].set_title(f'Image {j + 1}')

    for i, (clf_name, clf) in enumerate(clfs):
        cv = model_selection.cross_validate(
            clf,
            X,
            y=colors,
            scoring=scoring,
            cv=5
        )

        results = []

        for score_name in scoring:
            score = cv[f'test_{score_name}']

            best_clf_i = np.argmax(score)
            best_score = score[best_clf_i]

            avg_score = np.mean(score)

            obj = {}
            obj['score'] = score_name
            obj['best'] = f'{best_score:.3f}'
            obj['avg'] = f'{avg_score:.3f}'

            results.append(obj)

        print(f'[Image {j + 1}] {clf_name} => {results}')

        colors_pred = clf.fit_predict(X)

        ax[i + 1, j].scatter(x, y, c=colors_pred)

        result_str = ''
        for result in results:
            result_str += result.__str__() + '\n'

        ax[i + 1, j].set_title(f'{clf_name} \n {result_str}')

plt.subplots_adjust(hspace=0.5, wspace=1)
plt.savefig('clustering.png')
