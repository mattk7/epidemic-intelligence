import json
import os
import plotly.io as pio

# Define the path to the netsi template JSON file
template_path = os.path.join(os.path.dirname(__file__), 'netsi.json')
with open(template_path, "r") as f:
    netsi_template = json.load(f)

# Register the template and expose it as an attribute
pio.templates["netsi"] = netsi_template
netsi = "netsi"
