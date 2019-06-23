import numpy as np
import alpha_vantage
from datetime import datetime
from bokeh.embed import server_document
import requests
from io import BytesIO
from zipfile import ZipFile
import json
import pandas as pd
from bokeh.io import curdoc
from bokeh.server.server import Server
from bokeh.models import ColumnDataSource, CategoricalColorMapper, NumeralTickFormatter, HoverTool, Button, CustomJS
from bokeh.models.ranges import Range1d
from bokeh.models.widgets import Select, Button, Div
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.palettes import Spectral5

def modify_doc(doc):
    Q_API_key = '2QQxWV_Ycg2ULirbUMxB' # Quandl
    AV_API_key = '567TRV8RTL728INO' # Alpha Vantage 
    # Download WIKI metadata from Quandl via API
    url_0 = 'https://www.quandl.com/api/v3/databases/WIKI/metadata?api_key='+Q_API_key
    response_0 = requests.get(url_0)
    # Unzip the bytes and extract the csv file into memory
    myzip = ZipFile(BytesIO(response_0.content)).extract('WIKI_metadata.csv')
    # Read the csv into pandas dataframe
    df_0 = pd.read_csv(myzip)
    # Clean up the name fields
    df_0['name'] = df_0['name'].apply(lambda s:s[:s.find(')')+1].rstrip())
    # Drop extraneous fields and reorder
    df_0 = df_0.reindex(columns=['name','code'])
    # Make widgets
    stock_picker = Select(title="Select a Stock",  value=df_0['name'][0], options=df_0['name'].tolist())
    year_picker = Select(title="Select a Year",  value="2018", options=[str(i) for i in range(2008,2019)], width=100)
    months = ["January","February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    month_picker = Select(title="Select a Month",  value="January", options=months, width=150)
    widgets = row(stock_picker, year_picker, month_picker)

    # Get data
    def get_data(AV_API_key, ticker):
        from alpha_vantage.timeseries import TimeSeries
        ts = TimeSeries(key=AV_API_key)
        data, metadata = ts.get_daily(ticker,'full')
        data_dict = {}
        for sub in data.values():         
            for key, value in sub.items():
                data_dict.setdefault(key[3:], []).append(float(value))
        data_dict['date'] = list(data.keys())
        df = pd.DataFrame.from_dict(data_dict)
        df['date'] = pd.to_datetime(df['date'])
        df = df.iloc[::-1].reset_index(drop=True)
        # do some finance things
        df['inc'] = df.close > df.open
        df['inc'] = df['inc'].apply(lambda bool: str(bool))
        
        def SMA(n, s):
            return [s[0]]*n+[np.mean(s[i-n:i]) for i in range(n,len(s))]

        def EMA(n, s):
            k = 2/(n+1)
            ema = np.zeros(len(s))
            ema[0] = s[0]
            for i in range(1, len(s)-1):
                ema[i] = k*s[i] + (1-k)*ema[i-1]
            return ema

        df['ema12'] = EMA(12, df['open'])
        df['ema26'] = EMA(26, df['open'])
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = EMA(9, df['macd'])
        df['zero'] = df['volume'].apply(lambda x: x*0)
        df['hist'] = df.macd - df.signal
        df['histc'] = df.macd > df.signal
        df['histc'] = df['histc'].apply(lambda bool: str(bool))
        return df

    # Make figure
    df = get_data(AV_API_key, 'A')
    data_dict = df.to_dict('series')
    source = ColumnDataSource(data=data_dict)
    # create a new plot with a datetime axis type
    p1 = figure(plot_height=400, x_axis_type="datetime", tools="xwheel_pan,xwheel_zoom,pan,box_zoom", active_scroll='xwheel_zoom')
    p2 = figure(plot_height=150, x_axis_type="datetime", x_range=p1.x_range, tools="xwheel_pan,xwheel_zoom,pan,box_zoom", active_scroll='xwheel_pan')
    p3 = figure(plot_height=250, x_axis_type="datetime", x_range=p1.x_range, tools="xwheel_pan,xwheel_zoom,pan,box_zoom", active_scroll='xwheel_pan')
    # create price glyphs and volume glyph
    p1O = p1.line(x='date', y='open', source=source, color=Spectral5[0], alpha=0.8, legend="OPEN")
    p1C = p1.line(x='date', y='close', source=source, color=Spectral5[1], alpha=0.8, legend="CLOSE")
    p1L = p1.line(x='date', y='low', source=source, color=Spectral5[4], alpha=0.8, legend="LOW")
    p1H = p1.line(x='date', y='high', source=source, color=Spectral5[3], alpha=0.8, legend="HIGH")   
    p1.line(x='date', y='ema12', source=source, color="magenta", legend="EMA-12")
    p1.line(x='date', y='ema26', source=source, color="black", legend="EMA-26")
    color_mapper = CategoricalColorMapper(factors=["True", "False"], palette=["green", "red"])
    p1.segment(x0='date', y0='high', x1='date', y1='low', color={'field': 'inc', 'transform': color_mapper}, source=source)
    width_ms = 12*60*60*1000 # half day in ms
    p1.vbar(x='date', width=width_ms, top='open', bottom='close', color={'field': 'inc', 'transform': color_mapper}, source=source)
    p2V = p2.varea(x='date', y1='volume', y2='zero', source=source, color="black", alpha=0.8)
    p3.line(x='date', y='macd', source=source, color="green", legend="MACD")
    p3.line(x='date', y='signal', source=source, color="red", legend="Signal")
    p3.vbar(x='date', top='hist', source=source, width=width_ms, color={'field': 'histc', 'transform': color_mapper}, alpha=0.5)
    # Add HoverTools to each line
    p1.add_tools(HoverTool(tooltips=[('Date','@date{%F}'),('Open','@open{($ 0.00)}'),('Close','@close{($ 0.00)}'),
                                     ('Low','@low{($ 0.00)}'),('High','@high{($ 0.00)}'),('Volume','@volume{(0.00 a)}')],
                           formatters={'date': 'datetime'},mode='mouse'))
    p2.add_tools(HoverTool(tooltips=[('Date','@date{%F}'),('Open','@open{($ 0.00)}'),('Close','@close{($ 0.00)}'),
                                     ('Low','@low{($ 0.00)}'),('High','@high{($ 0.00)}'),('Volume','@volume{(0.00 a)}')],
                           formatters={'date': 'datetime'},mode='mouse'))
    p3.add_tools(HoverTool(tooltips=[('Date','@date{%F}'),('EMA-12','@ema12{($ 0.00)}'),('EMA-26','@ema26{($ 0.00)}'),
                                     ('MACD','@macd{($ 0.00)}'),('Signal','@signal{($ 0.00)}')],
                           formatters={'date': 'datetime'},mode='mouse'))
    p1.toolbar.logo = None
    p2.toolbar.logo = None
    p3.toolbar.logo = None
    # Add legend
    p1.legend.orientation = 'horizontal'
    p1.legend.title = 'Daily Stock Price'
    p1.legend.click_policy="hide"
    p1.legend.location="top_left"
    p3.legend.orientation = 'horizontal'
    p3.legend.location="top_left"
    p3.legend.orientation = 'horizontal'
    p3.legend.title = 'Moving Average Convergence Divergence'
    p3.legend.location="top_left"
    # Add axis labels
    #p1.xaxis.axis_label = 'Date'
    #p3.xaxis.axis_label = 'Date'
    p2.xaxis.axis_label = 'Date'
    p1.yaxis.axis_label = 'Price ($USD/share)'
    p2.yaxis.axis_label = 'Volume (shares)'
    p3.yaxis.axis_label = 'Indicator ($USD)'
    # Add tick formatting
    p1.yaxis[0].formatter = NumeralTickFormatter(format="$0.00")
    p2.yaxis[0].formatter = NumeralTickFormatter(format="0.0a")
    p3.yaxis[0].formatter = NumeralTickFormatter(format="$0.00")
    p1.outline_line_width = 1
    p2.outline_line_width = 1
    p3.outline_line_width = 1
    # Activate tools
    #p1.toolbar.active_scroll = 'xwheel_zoom'
    #p2.toolbar.active_scrool = 'xwheel_pan'

    # Set up callbacks
    def update_data(attrname, old, new):
        # Get the current Select value
        ticker = df_0.loc[df_0['name'] == stock_picker.value, 'code'].iloc[0]
        print('ticker:', ticker)
        # Get the new data
        df = get_data(AV_API_key, ticker)
        dfi = df.set_index(['date'])
        data_dict = df.to_dict('series')
        data_dict = {k:data_dict[k][::-1] for k in data_dict.keys()}
        source.data = ColumnDataSource(data=data_dict).data
        
    def update_axis(attrname, old, new):
        # Get the current Select values
        source.data = ColumnDataSource(data=data_dict).data
        year = year_picker.value
        month = f'{months.index(month_picker.value) + 1:02d}'   
        start = datetime.strptime(f'{year}-{month}-01', "%Y-%m-%d")
        if month == '12':
            end = datetime.strptime(f'{str(int(year)+1)}-01-01', "%Y-%m-%d")
        else:
            end = datetime.strptime(f'{year}-{int(month)+1:02d}-01', "%Y-%m-%d")     
        p1.x_range.start = start
        p1.x_range.end = end
        dfi = df.set_index(['date'])
        p1.y_range.start = dfi.loc[end:start]['low'].min()*0.95
        p1.y_range.end = dfi.loc[end:start]['high'].max()*1.05
        p2.y_range.start = dfi.loc[end:start]['volume'].min()*0.95
        p2.y_range.end = dfi.loc[end:start]['volume'].max()*1.05
        p3.y_range.start = dfi.loc[end:start]['macd'].min()*0.75
        p3.y_range.end = dfi.loc[end:start]['macd'].max()*1.25

    # setup JScallback
    JS = '''
        clearTimeout(window._autoscale_timeout);

        var date = source.data.date,
            low = source.data.low,
            high = source.data.high,
            volume = source.data.volume,
            macd = source.data.macd,
            start = cb_obj.start,
            end = cb_obj.end,
            min1 = Infinity,
            max1 = -Infinity,
            min2 = Infinity,
            max2 = -Infinity,
            min3 = Infinity,
            max3 = -Infinity;

        for (var i=0; i < date.length; ++i) {
            if (start <= date[i] && date[i] <= end) {
                max1 = Math.max(high[i], max1);
                min1 = Math.min(low[i], min1);
                max2 = Math.max(volume[i], max2);
                min2 = Math.min(volume[i], min2);
                max3 = Math.max(macd[i], max3);
                min3 = Math.min(macd[i], min3);
            }
        }
        var pad1 = (max1 - min1) * .05;
        var pad2 = (max2 - min2) * .05;
        var pad3 = (max3 - min3) * .05;

        window._autoscale_timeout = setTimeout(function() {
            y1_range.start = min1 - pad1;
            y1_range.end = max1 + pad1;
            y2_range.start = min2 - pad2;
            y2_range.end = max2 + pad2;
            y3d = Math.max(Math.abs(min3 - pad3), Math.abs(max3 + pad3))
            y3_range.start = -y3d;
            y3_range.end = y3d;
        });
    '''
    callbackJS = CustomJS(args={'y1_range': p1.y_range, 'y2_range': p2.y_range, 'y3_range': p3.y_range, 'source': source}, code=JS)

    p1.x_range.js_on_change('start', callbackJS)
    p1.x_range.js_on_change('end', callbackJS)

    stock_picker.on_change('value', update_data)
    year_picker.on_change('value', update_axis)
    month_picker.on_change('value', update_axis)

    b = Button(label='Full History')
    b.js_on_click(CustomJS(args=dict(p1=p1, p2=p2), code="""
    p1.reset.emit()
    p2.reset.emit()
    """))
    c = column(Div(text="", height=8), b, width=100)

    # Set up layouts and add to document
    row1 = row(stock_picker, year_picker, month_picker, c, height=65, width=800, sizing_mode='stretch_width')
    row2 = row(p1, width=800, height=400, sizing_mode='stretch_width')
    row3 = row(p2, width=800, height=150, sizing_mode='stretch_width')
    row4 = row(p3, width=800, height=250, sizing_mode='stretch_width')
    layout = column(row1, row2, row4, row3, width=800, height=800, sizing_mode='scale_both')
    doc.add_root(layout)

modify_doc(curdoc())