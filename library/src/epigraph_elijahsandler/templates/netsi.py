import plotly.io as pio

# copying plotly default to make 'netsi' theme
pio.templates['netsi']  = pio.templates['plotly_white']

# customize title
pio.templates['netsi']['layout']['title'] = dict(x=.5, y=.95, yref='container', 
                                                 font=dict(color='black', size=28, weight=400),
                                                 subtitle=dict(font=dict(color='#373737', size=14, weight=200))
                                                )
pio.templates['netsi']['layout']['title']['automargin'] = True

# font
pio.templates['netsi']['layout']['font'] = dict(family=r'../fonts/Barlow-Regular.ttf', size=14, color='black')

# custom colors
sequential_color_ls = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
pio.templates['netsi']['layout']['colorscale']['sequential'] = \
[[(a/(len(sequential_color_ls)-1)), color] for a, color in enumerate(sequential_color_ls)]

diverging_color_ls = ['#003f5c', '#577187', '#9ca8b4', '#e2e2e2', '#d0a2af', '#b8637e', '#9a1050']
pio.templates['netsi']['layout']['colorscale']['diverging'] = \
[[(a/(len(diverging_color_ls)-1)), color] for a, color in enumerate(diverging_color_ls)]

pio.templates['netsi']['layout']['shapedefaults']['line']['color'] = '#428299'

# colorway from nicole
pio.templates['netsi']['layout']['colorway'] = ['#428299', '#67C4D3', '#F48A64', '#77C6B1', 
                                                '#F2D65F', '#80A4CE', '#CC9EB1', '#BFD88F', '#8E8E8E']

netsi = pio.templates['netsi']