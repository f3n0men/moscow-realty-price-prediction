# moscow-realty-price-prediction

Предсказание цен на квартиры в Москве. Обучил Random Forest на реальных данных (R² = 0.871), завернул в веб-приложение — можно ввести параметры квартиры и получить оценку стоимости.

## Что сделано

- разведочный анализ данных (EDA) в ноутбуке
- отбор и подготовка признаков: площадь, район, станция метро и др.
- обучение Random Forest, R² = 0.871 на тестовой выборке
- веб-приложение для предсказания цены по параметрам квартиры

## Стек

- Python
- Pandas, NumPy — обработка данных
- Scikit-learn — модель и метрики
- Flask — веб-приложение
- Matplotlib — визуализация в ноутбуке

## Файлы

```
real_estate_analysis.ipynb       — EDA и обучение модели
data.csv                         — датасет
prediction_function.py           — логика предсказания
app.py                           — Flask-приложение
index.html                      — HTML шаблон
```

## Запуск

```bash
git clone https://github.com/f3n0men/moscow-realty-price-prediction.git
cd moscow-realty-price-prediction
pip install flask scikit-learn pandas numpy
python app.py
```

Открой `http://localhost:5000` в браузере.
