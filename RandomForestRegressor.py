import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

# 1. Carregar dataset
data_path = os.path.join("data", "spotify_songs.csv")
print("Lendo dataset em:", data_path)
df = pd.read_csv(data_path, on_bad_lines="skip")

# 🔧 Normalizar colunas
df.columns = df.columns.str.strip().str.lower()
print("Colunas detectadas:", df.columns.tolist())

# 2. Features numéricascolumns
features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "tempo", "duration_ms", "track_popularity"
]

# garantir que só use colunas que existem
features = [f for f in features if f in df.columns]

X = df[features]
y = df["playlist_subgenre"]

# 3. Pré-processamento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Modelo (Random Forest)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 6. Avaliação
y_pred = model.predict(X_test)
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
