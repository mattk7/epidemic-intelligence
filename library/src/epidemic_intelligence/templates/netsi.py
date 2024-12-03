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
pio.templates['netsi']['layout']['font'] = dict(family=r'../fonts/Barlow-Regular.ttf, Barlow, Droid Sans, PT Sans Narrow, Arial Narrow, Arial', size=14, color='black')

# custom colors
sequential_color_ls = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
pio.templates['netsi']['layout']['colorscale']['sequential'] = \
[[(a/(len(sequential_color_ls)-1)), color] for a, color in enumerate(sequential_color_ls)]

diverging_color_ls = ['#003f5c', '#577187', '#9ca8b4', '#e2e2e2', '#d0a2af', '#b8637e', '#9a1050']
pio.templates['netsi']['layout']['colorscale']['diverging'] = \
[[(a/(len(diverging_color_ls)-1)), color] for a, color in enumerate(diverging_color_ls)]

pio.templates['netsi']['layout']['shapedefaults']['line']['color'] = '#428299'

# colorway from nicole
pio.templates['netsi']['layout']['colorway'] = [
    '#428299', '#F48A64', '#77C6B1', '#F2D65F', '#80A4CE', '#CC9EB1', '#BFD88F', '#8E8E8E',
    # second colors
    "#D86F6F", "#719DD1", "#ED9D66", "#87C6BE", "#B0D3A1", "#AA7289", "#EABB71",
    "#3F96AA", "#C3CE86", "#B1E0D0", "#94B7E0", "#EDD77A", "#4A9986", "#B4BAD8",
    "#CE6E57", "#61BDCE", "#626D89", "#E27959", "#B57FB0", "#DADD97", "#40708B",
    "#61A36F", "#7D88B5", "#61AF98", "#A6D8E0", "#629BA5", "#C1B6B2", "#7BBA8A"
]

netsi = pio.templates['netsi']

# alessandra's color palettes
## continuous color scales
greenish = ['#1B6874', '#358F93', '#6CC2BD', '#BCE4DF', '#E2F2F1']
def get_greenish(level):
    assert level < len(greenish), f'level must be between 0 and {len(greenish)-1}'
    return ['#FFFFFF', greenish[level]]

## diverging
marton = ['#FFA982','#FFCE87','#FFE79C','#FFFFB7','#FFFFD5','#E5FFD4','#C4E3D0','#A7C6CE','#7F9ECE']