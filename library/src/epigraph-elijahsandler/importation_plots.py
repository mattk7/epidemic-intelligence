from google.cloud import bigquery
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from epigraph_elijahsandler.templates import netsi

def execute(client, query: str) -> pd.DataFrame:
    """Executes a BigQuery SQL query and returns the result as a DataFrame."""
    query_job = client.query(query)
    return query_job.to_dataframe()

def build_geographic_filter(geo_level: str, geo_values, alias: str = "g_target") -> str:
    """Builds a geographic filter based on the provided level and values."""
    if geo_values is not None:  # Only filter if geo_values is provided
        if isinstance(geo_values, list):
            if isinstance(geo_values[0], int):
                values = ', '.join(str(val) for val in geo_values)  # For INT64
                return f"{alias}.{geo_level} IN ({values})"
            elif isinstance(geo_values[0], str):
                values = ', '.join(f"'{val}'" for val in geo_values)  # For STRING
                return f"{alias}.{geo_level} IN ({values})"
        else:
            if isinstance(geo_values, int):
                return f"{alias}.{geo_level} = {geo_values}"
            elif isinstance(geo_values, str):
                return f"{alias}.{geo_level} = '{geo_values}'"
    return ""  # Return empty string if no filtering is needed

def build_ap_query(table_name: str, reference_table_name: str, source_geo_level: str, 
                   target_geo_level: str, output_geo_level: str = None, 
                   source_values=None, target_values=None, domestic: bool = True, 
                   cutoff: float = 0.05) -> str:
    """Builds an SQL query for analyzing importation data."""
    source_filter = build_geographic_filter(source_geo_level, source_values, alias="g_source") if source_values else 'TRUE'
    target_filter = build_geographic_filter(target_geo_level, target_values, alias="g_target") if target_values else 'TRUE'
    
    where_clauses = [target_filter, source_filter]
    
    if not domestic:
        where_clauses.append(f"g_source.{target_geo_level} <> g_target.{target_geo_level}")

    where_clause = ' AND '.join(where_clauses)

    query = f"""
    WITH region_imports AS (
      SELECT 
        g_source.{output_geo_level} AS source_label, 
        SUM(i.importations) AS total_importations
      FROM 
        `{table_name}` AS i
      JOIN 
        `{reference_table_name}` AS g_source 
        ON g_source.basin_id = i.source_basin
      JOIN 
        `{reference_table_name}` AS g_target 
        ON g_target.basin_id = i.target_basin
      WHERE 
        {where_clause}  
      GROUP BY 
        g_source.{output_geo_level}
    ),
    total_imports AS (
      SELECT 
        SUM(total_importations) AS grand_total_importations 
      FROM region_imports
    ),
    categorized_regions AS (
      SELECT 
        r.source_label,
        CASE 
          WHEN r.total_importations < ({cutoff} * (SELECT grand_total_importations FROM total_imports)) THEN 'Other'
          ELSE r.source_label
        END AS categorized_label
      FROM 
        region_imports r
    )
    SELECT 
      cr.categorized_label AS source, 
      i.date, 
      SUM(i.importations) AS importations,
      AVG(SUM(i.importations)) OVER (
        PARTITION BY cr.categorized_label 
        ORDER BY i.date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
      ) AS rolling_importations
    FROM 
      `{table_name}` AS i
    JOIN 
      `{reference_table_name}` AS g_target 
      ON g_target.basin_id = i.target_basin
    JOIN 
      `{reference_table_name}` AS g_source 
      ON g_source.basin_id = i.source_basin
    JOIN 
      categorized_regions cr 
      ON cr.source_label = g_source.{output_geo_level}
    WHERE 
      {where_clause}
    GROUP BY 
      cr.categorized_label, 
      i.date
    ORDER BY 
      i.date;
    """
    return query

def create_area_plot(data: pd.DataFrame, value: str, title: str = 'Area Plot', 
                     xlabel: str = 'Date', ylabel: str = 'Exportations', 
                     legendlabel: str = 'Source') -> go.Figure:
    """Creates an area plot using the provided data."""
    fig = px.area(data, x='date', y=value, color='source', template=netsi)
    fig.update_traces(line=dict(width=0.4))
    fig.update_layout(title_text=title, legend_traceorder='reversed',
                      xaxis_title=xlabel, yaxis_title=ylabel, legend_title_text=legendlabel)
    return fig

def area_plot(client, table_name: str, reference_table_name: str,
                                 source_geo_level: str, target_geo_level: str,
                                 output_geo_level: str = None, 
                                 source_values=None, target_values=None, 
                                 domestic: bool = True, cutoff: float = 0.05,
                                 value: str = 'importations', title: str = 'Area Plot',
                                 xlabel: str = 'Date', ylabel: str = 'Exportations', 
                                 legendlabel: str = 'Source') -> go.Figure:
    """Creates an area plot by executing a query based on the provided parameters."""
    
    # Step 1: Build the query
    query = build_ap_query(
        table_name=table_name,
        reference_table_name=reference_table_name,
        source_geo_level=source_geo_level,
        target_geo_level=target_geo_level,
        output_geo_level=output_geo_level,
        source_values=source_values,
        target_values=target_values,
        domestic=domestic,
        cutoff=cutoff
    )
    
    # Step 2: Execute the query
    data = execute(client, query)
    
    # Step 3: Create the area plot
    fig = create_area_plot(data, value=value, title=title, xlabel=xlabel, ylabel=ylabel, legendlabel=legendlabel)
    
    return fig
