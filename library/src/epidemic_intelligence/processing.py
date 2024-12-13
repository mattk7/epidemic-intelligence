from epidemic_intelligence.helper import execute, build_categorical_filter, generate_random_hash
import pandas as pd
from epiweeks import Week
from datetime import date as ddate

def summarize_runs(client, table_name, destination, 
                         method='median', 
                         source_column = 'source_basin', target_column = 'target_basin', 
                         value_column = 'importations', compartment_column = 'compartment'):
    

    if method in {'ftvariance', 'mbd', 'directional'}:
        # directional query
        directional = f"""WITH run_direction AS (
        SELECT
            run_id,
            date,
            {value_column},
            CASE
                WHEN {value_column} > LAG({value_column}) OVER(PARTITION BY run_id, {source_column}, {target_column}, {compartment_column} ORDER BY date) THEN 1
                WHEN {value_column} < LAG({value_column}) OVER(PARTITION BY run_id, {source_column}, {target_column}, {compartment_column} ORDER BY date) THEN -1
                ELSE 0
            END AS direction
        FROM
            `{table_name}`
        ), median_run_id AS (
            SELECT
                run_id,
                VARIANCE(direction) AS direction_variance
            FROM run_direction
            WHERE direction IS NOT NULL
            GROUP BY run_id
            ORDER BY direction_variance ASC
            LIMIT 1
            )
                """
        
        # mbd query
        mbd = f"""
        WITH cume_dist AS (
            SELECT 
                run_id, 
                date,
                CUME_DIST() OVER(PARTITION BY date, {source_column}, {target_column}, {compartment_column} ORDER BY {value_column} ASC) AS rank_asc,
                CUME_DIST() OVER(PARTITION BY date, {source_column}, {target_column}, {compartment_column} ORDER BY {value_column} DESC) AS  rank_desc
            FROM `{table_name}`
        ), median_run_id AS (
            SELECT
                run_id,
                SUM(rank_asc) * SUM(rank_desc) as mbd
            FROM cume_dist
            GROUP BY run_id
            ORDER BY mbd DESC
            LIMIT 1
        )
        """
        
        # fixed-time variance subquery
        ftvariance = f"""WITH run_variability AS (
            SELECT
                run_id,
                VARIANCE({value_column}) AS run_variance
            FROM
                `{table_name}`
            GROUP BY
                run_id, {source_column}, {target_column}, {compartment_column}
        ), median_run_id AS (
            SELECT
                run_id,
                SUM(run_variance) AS sum_run_variance
            FROM
                run_variability
            GROUP BY run_id
            ORDER BY
                sum_run_variance ASC
            LIMIT 1 )
                """
        
        medians = {'ftvariance': ftvariance, 'mbd': mbd, 'directional':directional}
        cent_query = f"""
        CREATE OR REPLACE TABLE `{destination}` AS
        {medians[method]}
        SELECT 
            run_id,
            {source_column},
            {target_column}, 
            compartment, 
            {value_column}, 
            date
        FROM
            `{table_name}`
        WHERE
            run_id = (SELECT run_id FROM median_run_id);
        """

        execute(client, cent_query)
        return True
    
    elif method == 'mean':
        mean_query = f"""
        CREATE OR REPLACE TABLE `{destination}` AS 
        SELECT 
            {source_column},
            {target_column}, 
            {compartment_column}, 
            AVG({value_column}) AS {value_column}, 
            date
        FROM
            `{table_name}`
        GROUP BY run_id, 
            {source_column},
            {target_column}, 
            {compartment_column},
            date
        """

        execute(client, mean_query)
        return True

    elif method == 'median':
        median_query = f"""
        CREATE OR REPLACE TABLE `{destination}` AS
        SELECT 
            {source_column},
            {target_column}, 
            {compartment_column}, 
            PERCENTILE_CONT({value_column}, 0.5) OVER (PARTITION BY {source_column}, {target_column}, {compartment_column}, date) AS {value_column}, 
            date
        FROM
            `{table_name}`
        """

        execute(client, median_query)
        return True
    
    else:
        print('''Please select a valid method: median, mean, mbd, ftvariance, or directional.
              Descriptions of these methods can be found in the epidemic-intelligence documentation.''')
        
def aggregate_importations(client, table_name, destination, 
                    source_column = 'source_basin', target_column = 'target_basin', 
                    value_column = 'importations',  run_id_column = None,
                    compartment_column = 'compartment',
                    compartments = False, new_compartment = 'compartment',
                    date='date'):
    cat_filter = build_categorical_filter(compartments, category_col=compartment_column, alias='t') if compartments is not False else "TRUE"
    where_clauses = [cat_filter]
    where_clause = " AND ".join(where_clauses)

    if date in ['date', 'iso']:
        agg_query = f"""
        CREATE OR REPLACE TABLE `{destination}` AS
        SELECT
            {f'{run_id_column} AS run_id, ' if run_id_column is not None else ''}
            {source_column},
            {target_column}, 
            {f"'{new_compartment}' AS compartment," if isinstance(compartments, list) or compartments==True else f't.{compartment_column} AS compartment,'}
            SUM({value_column}) as {value_column}, 
            {"CAST(EXTRACT(ISOYEAR FROM date) AS STRING) || 'W' || LPAD(CAST(EXTRACT(ISOWEEK FROM date) AS STRING), 2, '0')" if date=='iso' else 't.date'} AS date
        FROM
            `{table_name}` AS t
        WHERE
            {where_clause}
        GROUP BY
            {'run_id, ' if run_id_column is not None else ''}
            {source_column},
            {target_column}, 
            compartment, 
            date
        """
        execute(client, agg_query)

    elif date == 'epi':
        epiweek_dict = {}
        for year in range(1950, 2100):
            for month in range(1, 13):
                for day in range (1, 32):
                    try:
                        # epiweek_dict[f'{year:04d}-{month:02d}-{day:02d}'] = Week.fromdate(ddate(year, month, day)).cdcformat()
                        epiweek_dict[ddate(year, month, day)] = Week.fromdate(ddate(year, month, day)).cdcformat()
                    except(ValueError):
                        pass

        tn = pd.DataFrame(epiweek_dict, index=['epiweek']).T.reset_index(names='date')
        epitable = table_name.split('.')[0] + '.' + generate_random_hash()
        client.load_table_from_dataframe(tn, epitable).result()
        
        agg_query = f"""
            CREATE OR REPLACE TABLE `{destination}` AS
            SELECT
                t.{source_column},
                t.{target_column}, 
                {f"'{new_compartment}' AS compartment," if isinstance(compartments, list) or compartments==True else f't.{compartment_column}, '}
                SUM(t.{value_column}) as {value_column}, 
                e.epiweek AS date
            FROM
                `{table_name}` AS t
            JOIN
                `{epitable}` AS e ON e.date = t.date
            WHERE
                {where_clause}
            GROUP BY
                t.{source_column},
                t.{target_column}, 
                compartment, 
                date
            """
        
        execute(client, agg_query)
        client.delete_table(epitable, not_found_ok=True)

        return True


def aggregate_sir(client, table_name, destination, 
                    geo_column, value_columns, run_id_column='run_id',
                    date='epi'):
    """
    Aggregates data from a source table to a destination table based on specified logic.

    Args:
        client: Database client object to execute queries.
        table_name (str): Name of the source table.
        destination (str): Name of the destination table.
        geo_column (str): Column containing geographic data.
        value_columns (dict | list): Dictionary where keys are original columns and values are aggregated column names,
                                     or list of column names to aggregate with the same name.
        run_id_column (str): Name of the column containing run IDs. Defaults to 'run_id'.
        date (str): Date format to use ('iso' or 'epi'). Defaults to 'epi'.

    Returns:
        bool: True if operation succeeds, otherwise raises an error.
    """
    # Determine aggregation logic based on the type of value_columns
    if isinstance(value_columns, dict):
        # Prepare aggregation expressions for dictionary
        agg_mapping = {}
        for orig_col, agg_col in value_columns.items():
            if agg_col not in agg_mapping:
                agg_mapping[agg_col] = []
            agg_mapping[agg_col].append(orig_col)

        agg_values = ", ".join([
            f"SUM({'+'.join(columns)}) AS {agg_col}"
            for agg_col, columns in agg_mapping.items()
        ])

    elif isinstance(value_columns, list):
        # Prepare aggregation expressions for list (keep column names unchanged)
        agg_values = ", ".join([f"SUM({col}) AS {col}" for col in value_columns])

    else:
        raise ValueError("value_columns must be either a dictionary or a list.")

    if date == 'iso':
        agg_query = f"""
        CREATE OR REPLACE TABLE `{destination}` AS
        SELECT
            {run_id_column} AS run_id, 
            {geo_column} AS {geo_column}, 
            {agg_values},
            CAST(EXTRACT(ISOYEAR FROM t.date) AS STRING) || 'W' || 
            LPAD(CAST(EXTRACT(ISOWEEK FROM t.date) AS STRING), 2, '0') AS date
        FROM
            `{table_name}` AS t
        GROUP BY
            run_id,
            {geo_column},
            date
        """
        execute(client, agg_query)

    elif date == 'epi':
        epiweek_dict = {}
        for year in range(1950, 2100):
            for month in range(1, 13):
                for day in range(1, 32):
                    try:
                        epiweek_dict[ddate(year, month, day)] = Week.fromdate(ddate(year, month, day)).cdcformat()
                    except ValueError:
                        pass

        tn = pd.DataFrame(epiweek_dict, index=['epiweek']).T.reset_index(names='date')
        epitable = table_name.split('.')[0] + '.' + generate_random_hash()
        client.load_table_from_dataframe(tn, epitable).result()

        agg_query = f"""
        CREATE OR REPLACE TABLE `{destination}` AS
        SELECT
            {run_id_column} AS run_id, 
            {geo_column} AS {geo_column}, 
            {agg_values}, 
            e.epiweek AS date
        FROM
            `{table_name}` AS t
        JOIN
            `{epitable}` AS e ON e.date = t.date
        GROUP BY
            run_id,
            {geo_column},
            date
        """
        execute(client, agg_query)
        client.delete_table(epitable, not_found_ok=True)

    return True
