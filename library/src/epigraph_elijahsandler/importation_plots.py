from google.cloud import bigquery
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from epigraph_elijahsandler.templates import netsi
from epigraph_elijahsandler.helper import execute

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

def build_sankey_query(table_name, reference_table_name, source_geo_level, target_geo_level, source_values, target_values, date_range, 
                       cutoff=0.05, source_output_level=None, target_output_level=None, domestic=True):
                       
    if source_output_level == None:
        source_output_level = source_geo_level
    if target_output_level == None:
        target_output_level = target_geo_level
                       
    # Build filters for both source and target regions
    source_filter = build_geographic_filter(source_geo_level, source_values, alias="g_source")
    target_filter = build_geographic_filter(target_geo_level, target_values, alias="g_target")

    # Create the base where clause
    where_clauses = []

    if source_filter:
        where_clauses.append(source_filter)
    if target_filter:
        where_clauses.append(target_filter)
        
    if not domestic:
        # Exclude rows where target imports to itself
        where_clauses.append(f"g_source.{target_output_level} != g_target.{target_output_level}")

    # Join the where clauses with 'AND'
    where_clause = ' AND '.join(where_clauses)

    query = f"""
    WITH total_exportations AS (
        -- Calculate total exportations for the given date range
        SELECT 
            SUM(i.importations) AS total_sum
        FROM 
            `{table_name}` AS i
        JOIN 
            `{reference_table_name}` AS g_target 
            ON g_target.basin_id = i.target_basin
        JOIN 
            `{reference_table_name}` AS g_source 
            ON g_source.basin_id = i.source_basin
        WHERE 
            {where_clause}
            AND i.date >= '{date_range[0]}'
            AND i.date <= '{date_range[1]}'
    ), source_totals AS (
        -- Calculate total exportations for each source
        SELECT
            g_source.{source_output_level.split('_')[0]+'_id'} * -1 AS sourceid,
            g_source.{source_output_level} AS source,
            SUM(i.importations) AS source_sum
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
            AND i.date >= '{date_range[0]}'
            AND i.date <= '{date_range[1]}'
        GROUP BY sourceid, source
    ), target_totals AS (
        -- Calculate total exportations for each target
        SELECT
            g_target.{target_output_level.split('_')[0]+'_id'} AS targetid,
            g_target.{target_output_level} AS target,
            SUM(i.importations) AS target_sum
        FROM 
            `{table_name}` AS i
        JOIN 
            `{reference_table_name}` AS g_target
            ON g_target.basin_id = i.target_basin
        JOIN 
            `{reference_table_name}` AS g_source
            ON g_source.basin_id = i.source_basin
        WHERE 
            {where_clause}
            AND i.date >= '{date_range[0]}'
            AND i.date <= '{date_range[1]}'
        GROUP BY targetid, target
    ), categorized_sources AS (
        -- Categorize sources contributing less than the cutoff as "Other"
        SELECT 
            st.sourceid,
            CASE 
                WHEN st.source_sum < {cutoff} * t.total_sum THEN -1.5
                ELSE st.sourceid
            END AS revisedsourceid,
            CASE 
                WHEN st.source_sum < {cutoff} * t.total_sum THEN 'Other'
                ELSE st.source
            END AS source
        FROM 
            source_totals st
        CROSS JOIN 
            total_exportations t
    ), categorized_targets AS (
        -- Categorize targets contributing less than the cutoff as "Other"
        SELECT 
            tt.targetid,
            CASE 
                WHEN tt.target_sum < {cutoff} * t.total_sum THEN 1.5
                ELSE tt.targetid
            END AS revisedtargetid,
            CASE 
                WHEN tt.target_sum < {cutoff} * t.total_sum THEN 'Other'
                ELSE tt.target
            END AS target
        FROM 
            target_totals tt
        CROSS JOIN 
            total_exportations t
    ), final_exportations AS (
        -- Recalculate exportations with categorized sources and targets
        SELECT
            cs.sourceid,
            cs.revisedsourceid,
            cs.source,
            ct.targetid,
            ct.revisedtargetid,
            ct.target,
            SUM(i.importations) AS region_sum
        FROM 
            `{table_name}` AS i
        JOIN 
            `{reference_table_name}` AS g_source
            ON g_source.basin_id = i.source_basin
        JOIN 
            `{reference_table_name}` AS g_target
            ON g_target.basin_id = i.target_basin
        JOIN 
            categorized_sources cs
            ON cs.sourceid = g_source.{source_output_level.split('_')[0]+'_id'} * -1
        JOIN 
            categorized_targets ct
            ON ct.targetid = g_target.{target_output_level.split('_')[0]+'_id'}
        WHERE 
            {where_clause}
            AND i.date >= '{date_range[0]}'
            AND i.date <= '{date_range[1]}'
        GROUP BY 
            cs.sourceid, 
            cs.revisedsourceid,
            cs.source, 
            ct.targetid, 
            ct.revisedtargetid,
            ct.target
    )
    -- Final query to return exportations, ensuring "Other" sources and targets are properly grouped
    SELECT
        fe.revisedsourceid AS sourceid,
        fe.source AS source,
        fe.revisedtargetid AS targetid,
        fe.target AS target,
        SUM(fe.region_sum) / (SELECT total_sum FROM total_exportations) AS exportations
    FROM 
        final_exportations fe
    GROUP BY 
        fe.revisedsourceid, 
        fe.source, 
        fe.revisedtargetid, 
        fe.target;
    """

    return query

def create_sankey_plot(data, title):
  # Create a set of unique node IDs from both sourceid and targetid
  unique_ids = set(data['sourceid']).union(set(data['targetid']))

  # Create mapping for indices
  dict_indices = {id_: idx for idx, id_ in enumerate(unique_ids)}

  # Create mapping for labels (using the first occurrence of each name)
  name_mapping = {}
  for idx, row in data.iterrows():
      name_mapping[row['sourceid']] = row['source']
      name_mapping[row['targetid']] = row['target']

  # Generate source, target, and value lists for the Sankey diagram
  source = data['sourceid'].map(dict_indices)
  target = data['targetid'].map(dict_indices)
  value = data['exportations']

  # Create Sankey diagram
  fig = go.Figure(go.Sankey(
      node=dict(
          pad=15,
          thickness=20,
          line=dict(color='black', width=0.3),
          label=[name_mapping[id_] for id_ in dict_indices.keys()],  # Use names as node labels
      ),
      link=dict(
          source=source,  # Use mapped source indices
          target=target,  # Use mapped target indices
          value=value,
      )
  ))

  fig.update_layout(
      title_text = title,
      template=netsi
      )

  return fig

def sankey(client, table_name, reference_table_name, source_geo_level, target_geo_level, source_values, target_values, date_range, 
           cutoff=0.05, source_output_level=None, target_output_level=None, domestic=True, title="Sankey Diagram"):
    
    # Generate the query
    query = build_sankey_query(
        table_name=table_name,
        reference_table_name=reference_table_name,
        source_geo_level=source_geo_level,
        target_geo_level=target_geo_level,
        source_values=source_values,
        target_values=target_values,
        date_range=date_range,
        cutoff=cutoff,
        source_output_level=source_output_level,
        target_output_level=target_output_level,
        domestic=domestic
    )
    
    # Execute the query to get the data
    data = execute(client, query)
    
    # Create and return the Sankey plot
    return create_sankey_plot(data, title)

def build_bar_query(table_name, reference_table_name, source_geo_level, target_geo_level, source_values, target_values, date_range, 
                       cutoff=0.05, target_output_level=None, domestic=True):
    if target_output_level == None:
        target_output_level = target_geo_level
                       
    # Build filters for both source and target regions
    source_filter = build_geographic_filter(source_geo_level, source_values, alias="g_source")
    target_filter = build_geographic_filter(target_geo_level, target_values, alias="g_target")

    # Create the base where clause
    where_clauses = []

    if source_filter:
        where_clauses.append(source_filter)
    if target_filter:
        where_clauses.append(target_filter)
            
    if not domestic:
        # Exclude rows where target imports to itself
        where_clauses.append(f"g_source.{target_output_level} != g_target.{target_output_level}")
        
    # Join the where clauses with 'AND'
    where_clause = ' AND '.join(where_clauses)
        
    query = f"""
        WITH total_exportations AS (
            -- Calculate total exportations for the given date range
            SELECT 
                SUM(i.importations) AS total_sum
            FROM 
                `{table_name}` AS i
            JOIN 
                `{reference_table_name}` AS g_target 
                ON g_target.basin_id = i.target_basin
            JOIN 
                `{reference_table_name}` AS g_source 
                ON g_source.basin_id = i.source_basin
            WHERE 
                {where_clause}
                AND i.date >= '{date_range[0]}'
                AND i.date <= '{date_range[1]}'
        ), 
        target_totals AS (
            -- Calculate total exportations for each target
            SELECT
                g_target.{target_output_level.split('_')[0]+'_id'} AS targetid,
                g_target.{target_output_level} AS target,
                SUM(i.importations) AS target_sum
            FROM 
                `{table_name}` AS i
            JOIN 
                `{reference_table_name}` AS g_target
                ON g_target.basin_id = i.target_basin
            JOIN 
                `{reference_table_name}` AS g_source
                ON g_source.basin_id = i.source_basin
            WHERE 
                {where_clause}
                AND i.date >= '{date_range[0]}'
                AND i.date <= '{date_range[1]}'
            GROUP BY targetid, target
        ), 
        categorized_targets AS (
            -- Categorize targets contributing less than the cutoff as "Other"
            SELECT 
                tt.targetid,
                CASE 
                    WHEN tt.target_sum < {cutoff} * t.total_sum THEN -1
                    ELSE tt.targetid
                END AS revisedtargetid,
                CASE 
                    WHEN tt.target_sum < {cutoff} * t.total_sum THEN 'Other'
                    ELSE tt.target
                END AS target,
                tt.target_sum
            FROM 
                target_totals tt
            CROSS JOIN 
                total_exportations t
        )
        -- Final query to sum importations for each target and group "Other" regions
        SELECT 
            ct.revisedtargetid AS targetid,
            ct.target AS target,
            SUM(ct.target_sum) AS total_importations
        FROM 
            categorized_targets ct
        GROUP BY 
            targetid, target
        ORDER BY 
            total_importations DESC;

        """
    
    return query

def create_bar_chart(data: pd.DataFrame, title: str = 'Relative Risk of Importation', 
                     xlabel: str = 'Relative Risk of Importation', ylabel: str = 'Geography'):
    fig = px.bar(
    data.iloc[:, :], x='total_importations', y='target', orientation='h', 
    labels={
        'target': 'Target',
        'exportations': 'Relative Risk of Importation'
    },
    template='netsi'
)

# Sort y-axis by exportations with "Other" fixed at the bottom
    fig.update_layout(
        yaxis={'categoryorder': 'array', 'categoryarray': ['Other'] + sorted(
            [x for x in data['target'].unique() if x != 'Other'],
            key=lambda target: data.loc[data['target'] == target, 'exportations'].sum(),
            reverse=False
        )},
        title={
            'text': title
            },
        showlegend=False
    )

    return fig

def relative_risk(client, table_name, reference_table_name, source_geo_level, target_geo_level, source_values, target_values, date_range, 
           cutoff=0.05, target_output_level=None, domestic=True, 
           title="Relative Risk of Importation", xlabel="Relative Risk of Importation", 
           ylabel=None):

    # Generate the query
    query = build_bar_query(
        table_name=table_name,
        reference_table_name=reference_table_name,
        source_geo_level=source_geo_level,
        target_geo_level=target_geo_level,
        source_values=source_values,
        target_values=target_values,
        date_range=date_range,
        cutoff=cutoff,
        target_output_level=target_output_level,
        domestic=domestic
    )
    
    # Execute the query to get the data
    data = execute(client, query)
    
    # Create and return the Sankey plot
    return create_bar_chart(data, title, xlabel, 
                            ylabel if ylabel is not None else target_output_level)