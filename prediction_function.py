
def predict_apartment_price(area, rooms, minutes_to_metro, floor, total_floors, metro_station=None):
    """
    Предсказывает цену квартиры в Москве
    """
    import pandas as pd
    import joblib
    import json
    import gzip  # Добавьте этот импорт

    # Загружаем сжатую модель
    with gzip.open('apartment_price_predictor.pkl.gz', 'rb') as f:
        model = joblib.load(f)
    
    # Загружаем данные о станциях
    with open('station_data.json', 'r', encoding='utf-8') as f:
        station_data = json.load(f)
    
    premium_stations = station_data['premium_stations']
    budget_stations = station_data['budget_stations']
    features = {
        'Area': area,
        'Number of rooms': rooms,
        'Minutes to metro': minutes_to_metro,
        'Floor': floor,
        'Number of floors': total_floors,
        'is_premium_station': 0,
        'is_budget_station': 0
    }

    if metro_station in premium_stations:
        features['is_premium_station'] = 1
    elif metro_station in budget_stations:
        features['is_budget_station'] = 1

    input_df = pd.DataFrame([features])
    return model.predict(input_df)[0]
