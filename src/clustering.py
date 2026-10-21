from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

METADATA_COLUMNS = {"image_path", "label"}

def prepare_features(dataframe):
    """Retorna somente as características numéricas usadas pelo K-Means."""
    feature_columns = [
        column for column in dataframe.columns
        if column not in METADATA_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("Nenhuma característica numérica foi encontrada.")
    return dataframe[feature_columns].astype(float), feature_columns

def fit_kmeans(train_dataframe, n_clusters=2, random_state=42):
    """Ajusta o escalonador e o K-Means usando apenas as imagens de treino."""
    features, feature_columns = prepare_features(train_dataframe)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(features)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    train_clusters = model.fit_predict(train_scaled)
    return scaler, model, feature_columns, train_clusters

def predict_clusters(dataframe, scaler, model, feature_columns):
    """Atribui cada imagem a um cluster já aprendido no conjunto de treino."""
    features = dataframe[feature_columns].astype(float)
    return model.predict(scaler.transform(features))
