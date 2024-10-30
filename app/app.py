import gradio as gr
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


model_path = "./models/model1.pkl"
scaler_path = "./data/scaler.pkl"

TRAIN_COLS = ['Order Date', 'Retail Price', 'Release Date', 'Shoe Size', 'Order Year', 'Order Month', 'Order Day', 'Order Day of Week', 'Release Year', 'Release Month', 'Release Day', 'Release Day of Week', 'Brand_Adidas', 'Brand_Nike', 'Sub-brand_Air Jordan', 'Sub-brand_Yeezy', 'Product Line_Air Max', 'Product Line_Air Presto', 'Product Line_Air VaporMax', 'Product Line_Blazer Mid', 'Product Line_Boost', 'Product Line_Flyknit', 'Product Line_Mercurial', 'Product Line_Retro', 'Product Line_Zoom Fly', 'Model_1', 'Model_350', 'Model_90', 'Model_97', 'Version_2pt0', 'Version_V2', 'Height_High', 'Height_Low', 'Collaboration_Off White', 'Color_Oxford Tan', 'Color_Eve', 'Color_AF100', 'Color_Hallows', 'Color_White', 'Color_Chicago', 'Color_Elemental', 'Color_Queen', 'Color_Black-White', 'Color_Copper', 'Color_Blue Tint', 'Color_Frozen', 'Color_Black', 'Color_Yellow', 'Color_Menta', 'Color_All', 'Color_Zebra', 'Color_Turtledove', 'Color_Moonrock', 'Color_Semi', 'Color_Total-Orange', 'Color_Air', 'Color_Sesame', 'Color_Pink', 'Color_Pirate Black', 'Color_Green', 'Color_Beluga', 'Color_Wolf Grey', 'Color_Force', 'Color_Grim Reaper', 'Color_University Blue', 'Color_Black-Silver', 'Color_Desert Ore', 'Color_Red', 'Color_Cream White', 'Color_Core', 'Color_Rose', 'Color_Volt']

USA_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
    "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina",
    "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming"
]
BRANDS = ["Adidas", "Nike"]
SUB_BRANDS = [None, "Yeezy", "Air Jordan"]
PRODUCT_LINES = [None, "Boost", "Air Max", "Air Presto", "Air VaporMax", "Air Force 1", "Blazer Mid", "Zoom Fly",
                 "React Hyperdunk", "Hyperdunk", "Mercurial", "Retro", "Flyknit"]
MODELS = [None, 350, 1, 90, 97]
VERSIONS = [None, "V2", "2pto"]
HEIGHTS = [None, "High", "Low"]
COLLABORATIONS = [None, "off-white"]
COLORS = [
    None, 'Beluga', 'Core Black Copper', 'Core Black Green', 'Core Black Red', 'Core Black White',
    'Cream White', 'Zebra', 'Moonrock', 'Pirate Black', 'Oxford Tan', 'Turtledove',
    'Semi Frozen Yellow', 'Blue Tint', 'Black', 'Desert Ore', 'Elemental Rose Queen',
    'All Hallows Eve', 'Grim Reaper', 'Sesame', 'Wolf Grey', 'Menta', 'Black Silver',
    'Pink', 'Volt', 'Butter', 'Static', 'Static Reflective', 'Chicago', 'University Blue',
    'White', 'Black-White', 'Black-Silver', 'Total-Orange'
]

with open(model_path, 'rb') as file:
    clf = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)
    
def clean_dataset(data, TRAIN_COLS):
    numerical_cols = ['Retail Price', 'Shoe Size']
    data[numerical_cols] = scaler.transform(data[numerical_cols])
    
    df_final = pd.DataFrame(columns=TRAIN_COLS)

    data['Order Date'] = pd.to_datetime(data['Order Date'])
    data['Release Date'] = pd.to_datetime(data['Release Date'])

    df_final['Order Date'] = data['Order Date']
    df_final['Retail Price'] = data['Retail Price']
    df_final['Release Date'] = data['Release Date']
    df_final['Shoe Size'] = data['Shoe Size']

    df_final['Order Year'] = data['Order Date'].dt.year
    df_final['Order Month'] = data['Order Date'].dt.month
    df_final['Order Day'] = data['Order Date'].dt.day
    df_final['Order Day of Week'] = data['Order Date'].dt.dayofweek

    df_final['Release Year'] = data['Release Date'].dt.year
    df_final['Release Month'] = data['Release Date'].dt.month
    df_final['Release Day'] = data['Release Date'].dt.day
    df_final['Release Day of Week'] = data['Release Date'].dt.dayofweek

    df_final['Order Date'] = df_final['Order Date'].astype(np.int64) // 10**9
    df_final['Release Date'] = df_final['Release Date'].astype(np.int64) // 10**9

    if f'Brand_{data["Brand"][0]}' in df_final.columns:
        df_final[f'Brand_{data["Brand"][0]}'] = 1
    if f'Sub-brand_{data["Sub-brand"][0]}' in df_final.columns:
        df_final[f'Sub-brand_{data["Sub-brand"][0]}'] = 1
    if f'Product Line_{data["Product Line"][0]}' in df_final.columns:
        df_final[f'Product Line_{data["Product Line"][0]}'] = 1
    if f'Model_{data["Model"][0]}' in df_final.columns:
        df_final[f'Model_{data["Model"][0]}'] = 1
    if f'Version_{data["Version"][0]}' in df_final.columns:
        df_final[f'Version_{data["Version"][0]}'] = 1
    if f'Height_{data["Height"][0]}' in df_final.columns:
        df_final[f'Height_{data["Height"][0]}'] = 1
    if 'Collaboration_Off White' in df_final.columns:
        df_final['Collaboration_Off White'] = 1 if data['Collaboration'][0] == 'off-white' else 0

    if "Color(s)" in data.columns and data["Color(s)"].iloc[0]:
        color_list = data["Color(s)"].iloc[0].split(', ')
        
        for color in color_list:
            color_column = f'Color_{color}'
            if color_column in df_final.columns:
                df_final[color_column] = 1

    df_final = df_final.fillna(0)

    return df_final
    
def predict_shoe_data(
        retail_price, order_date, release_date, shoe_size, brand, sub_brand,
        product_line, model, version, height, collaboration, color
):
    if not retail_price or not release_date or not shoe_size or not brand:
        return "Error: Por favor completa todos los campos obligatorios."
    
    color = ', '.join(color) if color else None

    data = {
        "Retail Price": [retail_price],
        "Order Date": [order_date],
        "Release Date": [release_date],
        "Shoe Size": [shoe_size],
        "Brand": [brand],
        "Sub-brand": [sub_brand],
        "Product Line": [product_line],
        "Model": [model],
        "Version": [version],
        "Height": [height],
        "Collaboration": [collaboration],
        "Color(s)": [color],
    }
    df = pd.DataFrame(data)

    df_cleaned = clean_dataset(df,TRAIN_COLS)

    try:
        prediction = clf.predict(df_cleaned)
        return f"Predicción: ${prediction[0]:.2f}"
    except Exception as e:
        return f"Error en la predicción: {e}"

iface = gr.Interface(
    fn=predict_shoe_data,
    inputs=[
        gr.Number(label="Retail Price", precision=2),
        gr.Textbox(label="Order Date", value=str(datetime.today().date()), placeholder="YYYY-MM-DD"),
        gr.Textbox(label="Release Date", placeholder="YYYY-MM-DD"),
        gr.Number(label="Shoe Size"),
        gr.Dropdown(label="Brand", choices=BRANDS),
        gr.Dropdown(label="Sub-brand", choices=SUB_BRANDS),
        gr.Dropdown(label="Product Line", choices=PRODUCT_LINES),
        gr.Dropdown(label="Model", choices=MODELS),
        gr.Dropdown(label="Version", choices=VERSIONS),
        gr.Dropdown(label="Height", choices=HEIGHTS),
        gr.Dropdown(label="Collaboration", choices=COLLABORATIONS),
        gr.Dropdown(label="Color(s)", choices=COLORS, multiselect=True),
    ],
    outputs="text",
    title="Shoe Product Prediction Form",
    description="Complete los campos para predecir el resultado usando el modelo."
)

if __name__ == "__main__":
    iface.launch(inbrowser=True, server_port=7860, debug=True)
