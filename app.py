from flask import Flask, render_template, jsonify
import pandas as pd
import html

file_path = 'data/20231126.xlsx'
df = pd.read_excel(file_path)
df = df[df['Current Price'] >= 0]
df['Sector'] = df['Sector'].str.replace('/', ',')
df.drop(['Strengths', 'Limitations', "Today's High", "Today's Low"], axis=1, inplace=True)
df.rename(columns={'Overall Rating': 'Analysts Rating'}, inplace=True)

app = Flask(__name__)


def format_text_box(row, column_name):
    text_style = f"""background-color: {'green' if row[column_name] >= 0 else 'red'}; color: white; padding: 3px"""
    return f'<span style="{text_style}">{row[column_name]}</span>'


@app.route('/')
def index():
    return render_template('index.html')


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

    average_price_changes['Price Change %'] = average_price_changes['Price Change %'].round(2)
    average_market_cap['Market Cap in Crores'] = average_market_cap['Market Cap'].round(2)
    num_stocks_per_sector.columns = ['Sector', 'Number of Stocks']
    average_roe['ROE'] = average_roe['ROE'].round(2)
    average_pe['PE Ratio'] = average_pe['PE Ratio'].round(2)
    average_piotroski['Piotroski'] = average_piotroski['Piotroski'].round(2)
    average_de['Debt to Equity'] = average_de['Debt to Equity'].round(2)
    average_pb['PB Ratio'] = average_pb['PB Ratio'].round(2)
    average_analysts_rating['Analysts Rating'] = average_analysts_rating['Analysts Rating'].round(2)

    dataframes_to_merge = [
        average_price_changes,
        average_market_cap,
        num_stocks_per_sector,
        average_roe,
        average_pe,
        average_piotroski,
        average_de,
        average_pb,
        average_analysts_rating
    ]

    result_df = dataframes_to_merge[0]
    for df_to_merge in dataframes_to_merge[1:]:
        result_df = pd.merge(result_df, df_to_merge, on='Sector')

    result_df['Price Change %'] = result_df.apply(lambda row: format_text_box(row, 'Price Change %'), axis=1)

    return render_template('sector.html',
                           table=result_df[['Sector', 'Price Change %', 'Market Cap in Crores', 'Number of Stocks',
                                            'ROE', 'PE Ratio', 'Piotroski', 'Debt to Equity', 'PB Ratio', 'Analysts Rating']].to_html(
                               classes='table table-striped', index=False, escape=False, table_id='dataTable'))


@app.route('/selected_sector/<string:sector_name>', methods=['GET'])
def selected_sector(sector_name):
    sector_name_unescape = html.unescape(sector_name)
    selected_sector_df = df[df['Sector'] == sector_name_unescape]
    selected_sector_df.drop(columns=[
        'Sector', 'Ownership Rating', 'Financial Rating', 'Efficiency Rating', 'Valuation Rating'
    ], axis=1, inplace=True)

    selected_sector_df['SS Rating 1'] = round((selected_sector_df['Analysts Rating'] / 5) +
                                              (selected_sector_df['Piotroski'] / 9) * 2.5, 2)

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
