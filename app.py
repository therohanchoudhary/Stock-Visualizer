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


def format_text_box(row, column_name):
    text_style = f"""background-color: {'green' if row[column_name] >= 0 else 'red'}; color: white; padding: 3px"""
    return f'<span style="{text_style}">{row[column_name]}</span>'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommendation')
def stock_recommendation_selector():
    return render_template('stock_recommendation.html',
                           table=df.to_html(
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
    average_price_changes = df.groupby('Sector')['Price Change %'].mean().reset_index()
    average_market_cap = df.groupby('Sector')['Market Cap'].mean().reset_index()
    num_stocks_per_sector = df['Sector'].value_counts().reset_index()

    average_roe = df.groupby('Sector')['ROE'].mean().reset_index()
    average_pe = df.groupby('Sector')['PE Ratio'].mean().reset_index()
    average_piotroski = df.groupby('Sector')['Piotroski'].mean().reset_index()
    average_de = df.groupby('Sector')['Debt to Equity'].mean().reset_index()
    average_pb = df.groupby('Sector')['PB Ratio'].mean().reset_index()
    average_analysts_rating = df.groupby('Sector')['Analysts Rating'].mean().reset_index()
    average_ss_rating_1 = df.groupby('Sector')['SS rating 1'].mean().reset_index()

    average_price_changes['Price Change %'] = average_price_changes['Price Change %'].round(2)
    average_market_cap['Market Cap in Crores'] = average_market_cap['Market Cap'].round(2)
    num_stocks_per_sector.columns = ['Sector', 'Number of Stocks']
    average_roe['ROE'] = average_roe['ROE'].round(2)
    average_pe['PE Ratio'] = average_pe['PE Ratio'].round(2)
    average_piotroski['Piotroski'] = average_piotroski['Piotroski'].round(2)
    average_de['Debt to Equity'] = average_de['Debt to Equity'].round(2)
    average_pb['PB Ratio'] = average_pb['PB Ratio'].round(2)
    average_analysts_rating['Analysts Rating'] = average_analysts_rating['Analysts Rating'].round(2)
    average_ss_rating_1['SS rating 1'] = average_ss_rating_1['SS rating 1'].round(2)

    dataframes_to_merge = [
        average_price_changes,
        average_market_cap,
        num_stocks_per_sector,
        average_roe,
        average_pe,
        average_piotroski,
        average_de,
        average_pb,
        average_analysts_rating,
        average_ss_rating_1,
    ]

    result_df = dataframes_to_merge[0]
    for df_to_merge in dataframes_to_merge[1:]:
        result_df = pd.merge(result_df, df_to_merge, on='Sector')

    result_df['Price Change %'] = result_df.apply(lambda row: format_text_box(row, 'Price Change %'), axis=1)

    return render_template('sector.html',
                           table=result_df[['Sector', 'Price Change %', 'Market Cap in Crores', 'Number of Stocks',
                                            'ROE', 'PE Ratio', 'Piotroski', 'Debt to Equity', 'PB Ratio',
                                            'Analysts Rating', 'SS rating 1']].to_html(
                               classes='table table-striped', index=False, escape=False, table_id='dataTable'))


@app.route('/selected_sector/<string:sector_name>', methods=['GET'])
def selected_sector(sector_name):
    sector_name_unescape = html.unescape(sector_name)
    selected_sector_df = df[df['Sector'] == sector_name_unescape]
    selected_sector_df.drop(columns=['Sector'], axis=1, inplace=True)

    selected_sector_df['Analysts Rating'] = selected_sector_df['Analysts Rating'].astype(float)
    selected_sector_df['Price Change %'] = selected_sector_df.apply(lambda row: format_text_box(row, 'Price Change %'),
                                                                    axis=1)
    selected_sector_df['Analysts Rating'] = selected_sector_df.apply(
        lambda row: format_text_box(row, 'Analysts Rating'), axis=1)

    return render_template('selected_sector.html',
                           table=selected_sector_df.to_html(
                               classes='table table-striped', index=False, escape=False, table_id='dataTable'),
                           sector_name=sector_name_unescape
                           )


if __name__ == '__main__':
    app.run(debug=True)
