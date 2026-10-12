import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import random

# Фиксируем seed для воспроизводимости
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Загрузка данных
train = pd.read_csv('train.csv')
test = pd.read_csv('public_test.csv')

# Определяем категориальные признаки
cat_features = ['gender', 'region', 'family_status', 'education', 
                'employment_type', 'smartphone_brand']

# Целевая переменная
target = 'accepted'

# Признаки (исключаем customer_id и целевую)
features = [col for col in train.columns if col not in ['customer_id', target]]

# Разделение на train/valid
X_train, X_valid, y_train, y_valid = train_test_split(
    train[features], train[target], test_size=0.2, random_state=SEED, stratify=train[target]
)

# Создаём Pool для CatBoost (автоматически обрабатывает категориальные и пропуски)
train_pool = Pool(X_train, y_train, cat_features=cat_features)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

# Инициализация модели
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='F1',
    random_seed=SEED,
    verbose=100,
    early_stopping_rounds=50
)

# Обучение
model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

# Оценка на валидации
valid_preds = model.predict(X_valid)
f1 = f1_score(y_valid, valid_preds)
print(f'Validation F1: {f1:.4f}')

# Сохранение модели
model.save_model(f'catboost_model_seed_{SEED}.cbm')
print(f'Model saved as catboost_model_seed_{SEED}.cbm')