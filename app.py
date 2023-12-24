import html

import matplotlib
import numpy as np
import pandas as pd
from flask import Flask, render_template
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

matplotlib.use("TkAgg")

file_path = 'data/20231223.xlsx'
df = pd.read_excel(file_path)
df = df[df['Current Price'] >= 0]
df['Sector'] = df['Sector'].str.replace('/', ',')
df['Sector'] = df['Sector'].str.rstrip()
df.drop(columns=['EPS Ratio'], inplace=True)
sector_df = None

df.drop([
    'Strengths', 'Limitations', "Today's High", "Today's Low",
    "Ownership Rating", "Financial Rating", "Efficiency Rating",
    "Valuation Rating"
], axis=1, inplace=True)

df.rename(columns={'Overall Rating': 'Analysts Rating'}, inplace=True)
word_embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
ndim = 2
pca_sector = PCA(n_components=ndim)
embeddings = word_embedding_model.encode(df['Sector'].tolist(), convert_to_tensor=True)
embeddings_ndim = pca_sector.fit_transform(embeddings)
sector_embedding_columns = (lambda x: [f'SectorDim{x + 1}' for x in range(ndim)])(0)
ratio_columns = ['Piotroski', 'PE Ratio', 'PB Ratio', 'Debt to Equity']

df[sector_embedding_columns] = embeddings_ndim

df['SS rating 1'] = ((df['Analysts Rating'] / 5) + (df['Piotroski'] / 9)) * 2.5

app = Flask(__name__)

threshold_ratios = {
    'pe': [10, 40],  # since lower pe value indicates better pe ratio
    'pb': [3, 10],  # same reason as pe
    'de': [0.5, 3],  # same reason as pe
    'roe': [30, 0],
    'ss_rating_1': [3.5, 1],
    'piotroski': [7, 3],
}


def round_agg_df_col(x):
    return round(x.mean(), 2)


def format_text_box(row, column_name, threshold=0):
    text_style = f"""background-color: {'green' if row[column_name] >= threshold else 'red'}; color: white; padding: 3px"""
    return f'<span style="{text_style}">{row[column_name]}</span>'


def format_ratio_box(row, column_name, limits, inverse=False):
    value = row[column_name]
    text_style = f"""padding: 3px; background-color: """
    red_bg = text_style + "red; color: white"
    green_bg = text_style + "green; color: white"
    yellow_bg = text_style + "yellow; color: black"

    if value < 0 or (inverse and value > limits[1]) or (not inverse and value < limits[1]):
        final_style = red_bg
    elif (inverse and value < limits[0]) or (not inverse and value >= limits[0]):
        final_style = green_bg
    else:
        final_style = yellow_bg

    return f'<span style="{final_style}">{row[column_name]}</span>'


@app.route('/')
def index():
    return render_template('index.html')


def format_df(data):
    data['Analysts Rating'] = data['Analysts Rating'].astype(float).copy()

    data.loc[:, 'Analysts Rating'] = data.apply(
        lambda row: format_text_box(row, 'Analysts Rating'), axis=1)

    data['SS rating 1'] = data['SS rating 1'].astype(float).round(2).copy()
    data.loc[:, 'SS rating 1'] = data.apply(
        lambda row: format_ratio_box(row, 'SS rating 1', threshold_ratios['ss_rating_1']),
        axis=1)

    data['Debt to Equity'] = data['Debt to Equity'].astype(float).round(2).copy()
    data.loc[:, 'Debt to Equity'] = data.apply(
        lambda row: format_ratio_box(row, 'Debt to Equity', threshold_ratios['de'], inverse=True),
        axis=1)

    data.loc[:, 'Price Change %'] = data.apply(
        lambda row: format_text_box(row, 'Price Change %'),
        axis=1)

    data.loc[:, 'Piotroski'] = data.apply(
        lambda row: format_ratio_box(row, 'Piotroski', threshold_ratios['piotroski']),
        axis=1)

    data.loc[:, 'PE Ratio'] = data.apply(
        lambda row: format_ratio_box(row, 'PE Ratio', threshold_ratios['pe'], inverse=True),
        axis=1)

    data.loc[:, 'PB Ratio'] = data.apply(
        lambda row: format_ratio_box(row, 'PB Ratio', threshold_ratios['pb'], inverse=True),
        axis=1)

    data.loc[:, 'ROE'] = data.apply(
        lambda row: format_ratio_box(row, 'ROE', threshold_ratios['roe']),
        axis=1)
    return data


recommend_df = format_df(df)


@app.route('/recommendation')
def stock_recommendation_selector():
    return render_template('stock_recommendation.html',
                           table=recommend_df.to_html(
                               classes='table table-striped', index=False, escape=False, table_id='dataTable'),
                           )


@app.route('/recommendation/<string:stock_name>', methods=['GET'])
def stock_recommendation(stock_name):
    stock_name_unescape = html.unescape(stock_name)
    target_stock = df[df['NSE_CODE'] == stock_name_unescape].drop(['NSE_CODE'], axis=1)

    similarity_columns = ratio_columns + sector_embedding_columns

    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(df[similarity_columns])
    target_stock_normalized = scaler.transform(target_stock[similarity_columns])
    similarity_scores = cosine_similarity(target_stock_normalized, df_normalized)
    similar_stock_indices = np.argsort(similarity_scores[0])[:-101:-1]
    similar_stocks = df.iloc[similar_stock_indices]
    similar_stocks['Similarity Percentage'] = (
            (similarity_scores[0][similar_stock_indices] ** 2.5) * 100
    ).round(2)
    similar_stocks = similar_stocks.sort_values(by='Similarity Percentage', ascending=False)
    similar_stocks = pd.concat(
        [similar_stocks['NSE_CODE'], similar_stocks['Similarity Percentage'], similar_stocks.iloc[:, 1:-1]], axis=1)

    return render_template('single_stock_recommendation.html',
                           table=similar_stocks.to_html(classes='table table-striped', index=False, escape=False,
                                                        table_id='dataTable'),
                           stock_name=stock_name_unescape)


@app.route('/sector')
def sector():
    global sector_df

    if sector_df is None:
        agg_funcs = {}
        for col in ['Price Change %', 'Analysts Rating', 'Market Cap', 'ROE', 'Piotroski', 'Debt to Equity',
                    'PE Ratio', 'PB Ratio', 'SS rating 1']:
            agg_funcs[col] = round_agg_df_col

        df_by_sector = df.groupby('Sector').agg(agg_funcs).reset_index()
        num_stocks_per_sector = df['Sector'].value_counts().reset_index()
        num_stocks_per_sector.columns = ['Sector', 'Number of Stocks']
        sector_df = format_df(pd.merge(df_by_sector, num_stocks_per_sector, on='Sector'))

    return render_template('sector.html',
                           table=sector_df[['Sector', 'Price Change %', 'Market Cap',
                                            'Number of Stocks',
                                            'ROE', 'PE Ratio', 'Piotroski', 'Debt to Equity', 'PB Ratio',
                                            'Analysts Rating', 'SS rating 1']].to_html(
                               classes='table table-striped', index=False, escape=False, table_id='dataTable'))


@app.route('/selected_sector/<string:sector_name>', methods=['GET'])
def selected_sector(sector_name):
    sector_name_unescape = html.unescape(sector_name)
    selected_sector_df = df[df['Sector'] == sector_name_unescape]
    selected_sector_df = selected_sector_df.drop(columns=['Sector'], axis=1)

    selected_sector_df = format_df(selected_sector_df)

    return render_template('selected_sector.html',
                           table=selected_sector_df.to_html(
                               classes='table table-striped', index=False, escape=False, table_id='dataTable'),
                           sector_name=sector_name_unescape
                           )


if __name__ == '__main__':
    app.run(port=8000, debug=True)
