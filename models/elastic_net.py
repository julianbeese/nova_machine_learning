"""
Modul für die Implementierung des ElasticNet-Modells im Auto-Preis-Vorhersage-Projekt.

Dieses Modul enthält spezifische Funktionen für das Training und die Optimierung
des ElasticNet-Regressionsmodells.
"""
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from src.model.evaluate_model import evaluate_model


def train_elastic_net(X_train, y_train, X_test, y_test, preprocessor, param_grid=None, log_transformed=False):
    """
    Trainiert ein ElasticNet-Regressionsmodell mit Hyperparameter-Optimierung.

    Args:
        X_train: Training-Features
        y_train: Training-Zielwerte
        X_test: Test-Features
        y_test: Test-Zielwerte
        preprocessor: Vorverarbeitungspipeline (ColumnTransformer)
        param_grid: Parameter-Grid für GridSearchCV (optional)
        log_transformed: Ob die Zielwerte logarithmisch transformiert wurden

    Returns:
        tuple: (best_model, metrics) - Bestes Modell und Evaluierungsmetriken
    """
    # Standard-Parameter-Grid, falls keines angegeben wurde
    if param_grid is None:
        param_grid = {
            'regressor__alpha': [0.01, 0.1, 1.0, 10.0],
            'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'regressor__max_iter': [1000, 3000]
        }

    # Pipeline mit ElasticNet erstellen
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet(random_state=42))
    ])

    # Cross-Validation erstellen
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Grid Search mit Cross-Validation
    print("Training des ElasticNet-Modells mit Grid Search...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1  # Alle verfügbaren Prozessoren verwenden
    )

    # Grid Search ausführen
    grid_search.fit(X_train, y_train)

    # Ergebnisse ausgeben
    print("Beste Parameter für ElasticNet:")
    print(grid_search.best_params_)
    print(f"Bester CV-Score (neg. MSE): {grid_search.best_score_:.4f}")

    # Bestes Modell evaluieren
    best_model = grid_search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test, log_transformed)

    return best_model, metrics


def run_elastic_net_pipeline(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols, log_transformed=False):
    """
    Führt die komplette Pipeline für das ElasticNet-Modell aus.

    Diese Funktion erstellt den Präprozessor, trainiert das ElasticNet-Modell
    und evaluiert es.

    Args:
        X_train: Training-Features
        y_train: Training-Zielwerte
        X_test: Test-Features
        y_test: Test-Zielwerte
        numeric_cols: Liste der numerischen Spalten
        categorical_cols: Liste der kategorischen Spalten
        log_transformed: Ob die Zielwerte logarithmisch transformiert wurden

    Returns:
        tuple: (best_model, metrics) - Bestes Modell und seine Metriken
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    # Präprozessor erstellen
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # ElasticNet-Modell trainieren
    return train_elastic_net(X_train, y_train, X_test, y_test, preprocessor, log_transformed=log_transformed)