import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_features(features, image_path, random_seed=0):
    # visualize
    feature_data = np.array([feature[0].detach().numpy() for feature in features])
    feature_label = np.array([feature[1].numpy() for feature in features])
    feature_embedding = TSNE(n_components=2, learning_rate=50,
                             init='random', random_state=random_seed)\
                            .fit_transform(np.array(feature_data))
    plt.figure(figsize=(25, 30))
    sns.scatterplot(x=feature_embedding[:, 0],
                    y=feature_embedding[:, 1],
                    hue=feature_label,
                    palette=sns.hls_palette(as_cmap=True),
                    legend='full')
    plt.savefig(image_path)
    plt.close()
