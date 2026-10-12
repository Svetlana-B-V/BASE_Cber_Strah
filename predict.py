import pandas as pd
from catboost import CatBoostClassifier

# Фиксированный seed (должен совпадать с train.py)
SEED = 42

# Загрузка тестовых данных
test = pd.read_csv('public_test.csv')

# Загрузка модели
model = CatBoostClassifier()
model.load_model(f'catboost_model_seed_{SEED}.cbm')

# Признаки (те же, что при обучении)
features = [col for col in test.columns if col != 'customer_id']

# Предсказание
preds = model.predict(test[features])

# Формирование submission
submission = pd.DataFrame({
    'customer_id': test['customer_id'],
    'accepted': preds.astype(int)
})

# Сохранение
submission.to_csv(f'submission_seed_{SEED}.csv', index=False)
print(f'Submission saved as submission_seed_{SEED}.csv')