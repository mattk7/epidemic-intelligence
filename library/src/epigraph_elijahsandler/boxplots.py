from epigraph_elijahsandler.helper import execute, create_dataset, generate_random_hash, build_geographic_filter, hex_to_rgba
from epigraph_elijahsandler.templates import netsi
from google.cloud import bigquery
import pandas as pd
import time
import plotly.graph_objects as go

def functional_boxplot(client, table, reference_table, target, 
                       org_geography, geography=None, geography_value=None, 
                       date_range=None, 
                       num_clusters=1, num_features=10, grouping_method='mse', centrality_method='mse', threshold=1.5,
                       dataset=None, delete_data=True):

    # make sure we have a dataset name
    if dataset == None:
        dataset = generate_random_hash()
        
    # create dataset if it doesn't already exist
    create_dataset(client, dataset)

    # get id of geo target
    geo_id = geography.split('_')[0]+'_id'

    # Step 1: Create initial data table
    query_base = f""" CREATE OR REPLACE TABLE `{dataset}.data` AS
    SELECT 
        t.date,
        g.{geography} AS geo, 
        g.{geo_id} AS geoid,
        t.run_id,
        SUM(t.{target}) as value
    FROM `{table}` as t
    JOIN `{reference_table}` AS g
        ON g.{org_geography} = t.{org_geography}
    WHERE 
        {build_geographic_filter(geography, geography_value, alias='g')}
        {f"AND t.date >= '{date_range[0]}' AND t.date <= '{date_range[1]}'" if date_range is not None else ''}
    --  AND run_id BETWEEN 1 AND 100
    GROUP BY date, geoid, geo, run_id
    ORDER BY date
    ;"""

    df = client.query(query_base).result()  # Execute the query to create the table
    print("Data sliced successfully.")

    # Step 2: Create the curve distance table for mse and abc
    query_distances = f"""
    CREATE OR REPLACE TABLE `{dataset}.curve_distances` AS
    SELECT
        a.run_id AS run_id_a,
        b.run_id AS run_id_b,
        AVG(POW(a.value - b.value, 2)) AS mse,
        SUM(ABS(a.value - b.value)) AS abc
    FROM
        `{dataset}.data` a
    JOIN
        `{dataset}.data` b
    ON
        a.date = b.date
    GROUP BY
        run_id_a, run_id_b
    """
    client.query(query_distances).result()  # Execute the query to create the table
    print("Curve distance table created successfully.")

    # Step 3: Create the distance matrix
    query_distance_matrix = f"""
    CREATE OR REPLACE TABLE `{dataset}.distance_matrix`
    CLUSTER BY run_id AS --optimizations
    SELECT
        run_id_a AS run_id,
            ARRAY_AGG(STRUCT(run_id_b, {grouping_method}) ORDER BY run_id_b ASC) AS distances
    FROM
        `{dataset}.curve_distances`
    GROUP BY
        run_id_a;
    """
    client.query(query_distance_matrix).result()  # Execute the query to create the table
    print("Distance matrix table created successfully.")

    # Step 4: Handle case when num_clusters = 1
    if num_clusters == 1:
        query_assign_all_to_one_cluster = f"""
        CREATE OR REPLACE TABLE `{dataset}.kmeans_results`
        CLUSTER BY CENTROID_ID, run_id AS
        SELECT DISTINCT
            run_id,
            1 AS centroid_id  -- Assign all runs to centroid 1
        FROM 
            `{dataset}.distance_matrix`
        """
        s = time.time()
        client.query(query_assign_all_to_one_cluster).result()  # Execute the query to assign all runs to centroid 1
        print(f"All runs assigned to centroid 1 successfully in {round(time.time() - s, 3)} seconds.")
        
    else:
        # Step 4: Create the K-means model by selecting the first num_features features based on actual distances
        query_create_model = f"""
        CREATE OR REPLACE MODEL `{dataset}.kmeans_model`
        OPTIONS(model_type='kmeans', num_clusters={num_clusters}) AS
        SELECT
            run_id,
            ARRAY(
                SELECT distance.{grouping_method} 
                FROM UNNEST(distances) AS distance 
                WHERE distance.run_id_b <= {num_features}  -- Select only the first num_features
            ) AS features
        FROM
            `{dataset}.distance_matrix`;
        """
        s = time.time()
        client.query(query_create_model).result()  # Execute the model creation
        print(f"K-means model created successfully in {round(time.time() - s, 3)} seconds.")

        # Step 5: Apply K-means clustering and save results in a table
        query_kmeans = f"""
        CREATE OR REPLACE TABLE `{dataset}.kmeans_results`
        CLUSTER BY CENTROID_ID, run_id AS
        SELECT
            *
        FROM
            ML.PREDICT(MODEL `{dataset}.kmeans_model`,
                (SELECT
                    run_id,
                    ARRAY(
                        SELECT distance.{grouping_method} 
                        FROM UNNEST(distances) AS distance 
                        WHERE distance.run_id_b <= {num_features}  
                    ) AS features
                FROM
                    `{dataset}.distance_matrix`)
            ) AS predictions
        """
        s = time.time()
        client.query(query_kmeans).result()  # Execute the model creation
        print(f"K-means clustering results saved successfully in {round(time.time() - s, 3)} seconds.")

    # Step 6: Get summed distances for abc and mse
    s = time.time()
    save_sum_distances = f"""CREATE OR REPLACE TABLE `{dataset}.total_distances_table` AS
        WITH a AS (
            SELECT 
                kr.CENTROID_ID,
                kr.run_id, 
                run_id_b, 
                {centrality_method}  
            FROM 
                `{dataset}.kmeans_results` AS kr
            JOIN 
                `{dataset}.curve_distances` AS dm
            ON 
                kr.run_id = dm.run_id_a
    --        CROSS JOIN 
    --            UNNEST(dm.distances) AS dm_dist  -- Unnest the distances array here 
        ),
        b AS (
            SELECT
                run_id AS run_id_b, 
                CENTROID_ID AS CENTROID_ID_B
            FROM
                `{dataset}.kmeans_results`
        )
        SELECT 
            a.run_id,
            a.CENTROID_ID,
            AVG({centrality_method}) AS total_distance  
        FROM 
            a
        JOIN 
            b
        ON 
            a.run_id_b = b.run_id_b
        WHERE
            a.CENTROID_ID = b.CENTROID_ID_B
        GROUP BY
            a.CENTROID_ID,
            a.run_id;
        """
    if centrality_method in ['abc', 'mse']:
        client.query(save_sum_distances).result()  # Execute the model creation
        print(f"Distance sum results using {centrality_method.upper()} saved successfully in {round(time.time()-s, 3)}.")

    # or do the same for mbd
    s = time.time()
    mbd = f"""
    CREATE OR REPLACE TABLE `{dataset}.total_distances_table`AS
    WITH curve_data AS (
        SELECT DISTINCT
            a.date AS date,
            a.run_id AS run_id,
            kra.CENTROID_ID as CENTROID_ID,
            b.run_id AS boundary_1_id,
            c.run_id AS boundary_2_id, 
            (a.value) AS curve,
            (b.value) AS boundary_1,
            (c.value) AS boundary_2
        FROM
            `{dataset}.data` AS a
        JOIN
            `{dataset}.data` AS b ON a.date = b.date
        JOIN
            `{dataset}.data` AS c ON a.date = c.date
        JOIN 
            `{dataset}.kmeans_results` AS kra ON a.run_id = kra.run_id
        JOIN 
            `{dataset}.kmeans_results` AS krb ON b.run_id = krb.run_id
        JOIN
            `{dataset}.kmeans_results` AS krc ON c.run_id = krc.run_id
        
        WHERE
            b.run_id < c.run_id
            AND a.run_id != b.run_id
            AND a.run_id != c.run_id
            AND kra.CENTROID_ID = krb.CENTROID_ID
            AND kra.CENTROID_ID = krc.CENTROID_ID

    --   GROUP BY
    --   a.date, a.run_id, CENTROID_ID, b.run_id, c.run_id
        ORDER BY
        a.run_id, b.run_id, c.run_id, a.date
    )


    SELECT
        run_id,
        CENTROID_ID,
        1 / COUNT(*) as total_distance
    FROM curve_data
    WHERE
    (curve_data.boundary_1 <= curve AND curve <= boundary_2)
    OR (curve_data.boundary_2 <= curve AND curve <= boundary_1)
    GROUP BY run_id, CENTROID_ID
    """
    # centrality_method = 'mbd'
    if centrality_method == 'mbd':
        df = client.query(mbd).result()  # Execute the query to create the table
        print(f"MBD calculated in {round(time.time()-s, 3)} seconds.")

    # saving and fetching data needed for visualizing
    # Step 7
    query_non_outliers = f"""
    CREATE OR REPLACE TABLE `{dataset}.non_outliers_table` AS
    WITH iqr_bounds AS (
    SELECT 
        CENTROID_ID,
        --Using approximate quantiles to save some time and space
        APPROX_QUANTILES(total_distance, 100)[OFFSET(25)] AS lower_quartile,
        APPROX_QUANTILES(total_distance, 100)[OFFSET(75)] AS upper_quartile
    FROM `{dataset}.total_distances_table`
    GROUP BY CENTROID_ID
    ),
    non_outliers AS (
    SELECT 
        d.CENTROID_ID,
        d.run_id
    FROM `{dataset}.total_distances_table` d
    JOIN iqr_bounds b
        ON d.CENTROID_ID = b.CENTROID_ID
    WHERE d.total_distance BETWEEN 
            (b.lower_quartile - {threshold} * (b.upper_quartile - b.lower_quartile)) 
            AND (b.upper_quartile + {threshold} * (b.upper_quartile - b.lower_quartile))
    )

    SELECT * FROM non_outliers;

        """
    client.query(query_non_outliers).result()  # Execute the model creation
    print(f"Non-outliers saved successfully.")



    query_middle_curves = f"""
    CREATE OR REPLACE TABLE `{dataset}.middle_curves` AS
    WITH grouped_data AS (
    SELECT 
        CENTROID_ID,
        run_id,
        total_distance,
        ROW_NUMBER() OVER (PARTITION BY CENTROID_ID ORDER BY total_distance) AS rn,
        COUNT(*) OVER (PARTITION BY CENTROID_ID) AS total_count
    FROM `{dataset}.total_distances_table`
    ),
    top_half AS (
    SELECT 
        CENTROID_ID,
        run_id,
        total_distance,
        rn
    FROM grouped_data
    WHERE rn <= CAST(total_count * 0.5 AS INT64)  -- Select the top 50% based on total_distance
    )

    SELECT 
        CENTROID_ID,
        run_id,
    FROM top_half;  -- Select the required fields for the middle_curves table

    """
    client.query(query_middle_curves).result()  # Execute the model creation
    print(f"Middle curves saved successfully.")

    save_median = f"""
    CREATE OR REPLACE TABLE `{dataset}.median_curves` AS
    WITH ranked_data AS (
    SELECT 
        CENTROID_ID,
        run_id,
        total_distance,
        ROW_NUMBER() OVER (PARTITION BY CENTROID_ID ORDER BY total_distance) AS rn
    FROM `{dataset}.total_distances_table`
    )

    SELECT 
        CENTROID_ID,
        run_id,
    FROM ranked_data
    WHERE rn = 1;  -- Select the run_id with the lowest total_distance for each CENTROID_ID
    """

    client.query(save_median).result()  # Execute the model creation
    print(f"Median curves saved successfully.")

    # Step 8
    get_median_curves = f"""-- Step 3: Calculate min and max values at each time step using the non-outliers table
        SELECT
            data.date,
            mc.CENTROID_ID,
            MAX(data.value) as median
        FROM
            `{dataset}.data` as data
        JOIN
            `{dataset}.median_curves` as mc
        ON
            data.run_id = mc.run_id
        GROUP BY
            date, 
            CENTROID_ID
        ORDER BY
            CENTROID_ID, 
            date;
        """
    plt_median = client.query(get_median_curves).to_dataframe()  # Execute and fetch results
    print("Curves extracted successfully.")

    # Step 8
    get_curves = f"""-- Step 3: Calculate min and max values at each time step using the non-outliers table
        
        SELECT
            data.date,
            nout.CENTROID_ID,
            MAX(data.value) as curve_100,
            MIN(data.value) as curve_0
        FROM
            `{dataset}.data` as data
        JOIN
            `{dataset}.non_outliers_table` as nout
        ON
            data.run_id = nout.run_id
        GROUP BY
            date, 
            CENTROID_ID
        ORDER BY
            CENTROID_ID, 
            date;
        """
    plt_curves = client.query(get_curves).to_dataframe()  # Execute and fetch results
    print("Curves extracted successfully.")

    # Step 8
    get_mid_curves = f"""-- Step 3: Calculate min and max values at each time step using the non-outliers table
        SELECT
            data.date,
            mc.CENTROID_ID,
            MAX(data.value) as curve_75,
            MIN(data.value) as curve_25
        FROM
            `{dataset}.data` as data
        JOIN
            `{dataset}.middle_curves` as mc
        ON
            data.run_id = mc.run_id
        GROUP BY
            date, 
            CENTROID_ID
        ORDER BY
            CENTROID_ID, 
            date;
        """
    plt_middle = client.query(get_mid_curves).to_dataframe()  # Execute and fetch results
    print("Curves extracted successfully.")

    get_outliers = f"""-- Step 3: Calculate min and max values at each time step using the non-outliers table
        WITH outliers AS (
            SELECT 
                tdt.run_id, 
                tdt.CENTROID_ID
            FROM 
                `{dataset}.total_distances_table` as tdt
            WHERE 
                run_id NOT IN (SELECT run_id FROM `{dataset}.non_outliers_table`)
        )
        
        
        SELECT
            data.date,
            outliers.CENTROID_ID,
            outliers.run_id,
            data.value
        FROM
            outliers
        JOIN
            `{dataset}.data` as data
        ON
            data.run_id = outliers.run_id
        ORDER BY
            run_id, 
            date;
        """
    plt_outliers = client.query(get_outliers).to_dataframe()  # Execute and fetch results
    print("Curves extracted successfully.")

    # Creating a central graphing df
    merged_curves = pd.merge(plt_curves, plt_middle, on=['date', 'CENTROID_ID'], how='inner')
    merged_curves = pd.merge(merged_curves, plt_median)

    # Graph!
    # create and lay out graph
    fig = go.Figure()
    fig.update_layout(
        title={
            'text': f"Functional Boxplot",
            },
        xaxis_title="Date",
        yaxis_title="Incidence",
        template=netsi,

    )

    colors = netsi.layout.colorway

    # plot outliers
    for run in plt_outliers['run_id'].unique():
            df_run = plt_outliers[plt_outliers['run_id'] == run]
            gr = df_run.iloc[0, 1] # careful not to change table formatting
        
            fig.add_trace(go.Scatter(
            name=f'Group {gr} Outlier',
            x=df_run['date'],
            y=df_run['value'],
            marker=dict(color=hex_to_rgba(colors[gr-1], alpha=.3)),
            line=dict(width=1, dash='solid'),
            mode='lines',
            showlegend=False,
            legendgroup=str(gr)  # Assign to legend group
        ))

    for group in plt_median['CENTROID_ID'].unique():
        print(group)
        # actually graph
        # Lower
        fig.add_trace(go.Scatter(
            name=f'Group {group} Lower Bound',
            x=merged_curves[merged_curves['CENTROID_ID'] == group]['date'],
            y=merged_curves[merged_curves['CENTROID_ID'] == group]['curve_0'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            showlegend=False,
            legendgroup=str(group)  # Assign to legend group
        ))
        # Upper
        fig.add_trace(go.Scatter(
            name=f'Group {group} Upper Bound',
            x=merged_curves[merged_curves['CENTROID_ID'] == group]['date'],
            y=merged_curves[merged_curves['CENTROID_ID'] == group]['curve_100'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            fillcolor=hex_to_rgba(colors[group-1], alpha=.3),
            fill='tonexty',
            showlegend=False,
            legendgroup=str(group)  # Assign to legend group
        ))
            
        # Lower
        fig.add_trace(go.Scatter(
            name=f'Group {group} Lower Quartile',
            x=merged_curves[merged_curves['CENTROID_ID'] == group]['date'],
            y=merged_curves[merged_curves['CENTROID_ID'] == group]['curve_25'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            showlegend=False,
            legendgroup=str(group)  # Assign to legend group
        ))
        # Upper
        fig.add_trace(go.Scatter(
            name=f'Group {group}',
            x=merged_curves[merged_curves['CENTROID_ID'] == group]['date'],
            y=merged_curves[merged_curves['CENTROID_ID'] == group]['curve_75'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            fillcolor=hex_to_rgba(colors[group-1], alpha=.3),
            fill='tonexty',
            showlegend=True,
            legendgroup=str(group)  # Assign to legend group
        ))
        
        
        fig.add_trace(go.Scatter(
            name=f'Group {group} Median',
            x=merged_curves[merged_curves['CENTROID_ID'] == group]['date'],
            y=merged_curves[merged_curves['CENTROID_ID'] == group]['median'],
            marker=dict(color=hex_to_rgba(colors[group-1], alpha=1)),
            line=dict(width=1),
            mode='lines',
            showlegend=False,
            legendgroup=str(group)  # Assign to legend group
        ))

    if delete_data:
        client.delete_dataset(
            dataset,
            delete_contents=True, 
            not_found_ok=True 
        )
        print(f"BigQuery dataset `{client.project}.{dataset}` removed successfully, or it did not exist.")

    return fig

def fixed_time_boxplot(client, table, reference_table, target, 
                       org_geography, geography=None, geography_value=None, 
                       date_range=None, 
                       num_clusters=1, num_features=10, grouping_method='mse', 
                       dataset=None, delete_data=True, kmeans_table=False,
                       confidence=.9, full_range = False, outlying_points = True):

    # get id of geo target
    geo_id = geography.split('_')[0]+'_id'

    # make sure we have a dataset name
    if dataset == None:
        dataset = generate_random_hash()
                
    # create dataset if it doesn't already exist
    create_dataset(client, dataset)

    # Step 1: Create initial data table
    query_base = f""" CREATE OR REPLACE TABLE `{dataset}.data` AS
            SELECT 
                t.date,
                g.{geography} AS geo, 
                g.{geo_id} AS geoid,
                t.run_id,
                SUM(t.{target}) as value
            FROM `{table}` as t
            JOIN `{reference_table}` AS g
                ON g.{org_geography} = t.{org_geography}
            WHERE 
                {build_geographic_filter(geography, geography_value, alias='g')}
                {f"AND t.date >= '{date_range[0]}' AND t.date <= '{date_range[1]}'" if date_range is not None else ''}
            --  AND run_id BETWEEN 1 AND 100
            GROUP BY date, geoid, geo, run_id
            ORDER BY date
            ;"""

    df = client.query(query_base).result()  # Execute the query to create the table
    print("Data sliced successfully.")

    if kmeans_table is False:
        if num_clusters > 1:

            # Step 2: Create the curve distance table for mse and abc
            query_distances = f"""
            CREATE OR REPLACE TABLE `{dataset}.curve_distances` AS
            SELECT
                a.run_id AS run_id_a,
                b.run_id AS run_id_b,
                AVG(POW(a.value - b.value, 2)) AS mse,
                SUM(ABS(a.value - b.value)) AS abc
            FROM
                `{dataset}.data` a
            JOIN
                `{dataset}.data` b
            ON
                a.date = b.date
            GROUP BY
                run_id_a, run_id_b
            """
            client.query(query_distances).result()  # Execute the query to create the table
            print("Curve distance table created successfully.")

            # Step 3: Create the distance matrix
            query_distance_matrix = f"""
            CREATE OR REPLACE TABLE `{dataset}.distance_matrix`
            CLUSTER BY run_id AS --optimizations
            SELECT
                run_id_a AS run_id,
                    ARRAY_AGG(STRUCT(run_id_b, {grouping_method}) ORDER BY run_id_b ASC) AS distances
            FROM
                `{dataset}.curve_distances`
            GROUP BY
                run_id_a;
            """
            client.query(query_distance_matrix).result()  # Execute the query to create the table
            print("Distance matrix table created successfully.")

        # Step 4: Handle case when num_clusters = 1
        if num_clusters == 1:
            query_assign_all_to_one_cluster = f"""
            CREATE OR REPLACE TABLE `{dataset}.kmeans_results`
            CLUSTER BY CENTROID_ID, run_id AS
            SELECT DISTINCT
                run_id,
                1 AS centroid_id  -- Assign all runs to centroid 1
            FROM 
                `{dataset}.data`
            """
            s = time.time()
            client.query(query_assign_all_to_one_cluster).result()  # Execute the query to assign all runs to centroid 1
            print(f"All runs assigned to centroid 1 successfully in {round(time.time() - s, 3)} seconds.")
            
        else:
            # Step 4: Create the K-means model by selecting the first num_features features based on actual distances
            query_create_model = f"""
            CREATE OR REPLACE MODEL `{dataset}.kmeans_model`
            OPTIONS(model_type='kmeans', num_clusters={num_clusters}) AS
            SELECT
                run_id,
                ARRAY(
                    SELECT distance.{grouping_method} 
                    FROM UNNEST(distances) AS distance 
                    WHERE distance.run_id_b <= {num_features}  -- Select only the first num_features
                ) AS features
            FROM
                `{dataset}.distance_matrix`;
            """
            s = time.time()
            client.query(query_create_model).result()  # Execute the model creation
            print(f"K-means model created successfully in {round(time.time() - s, 3)} seconds.")

            # Step 5: Apply K-means clustering and save results in a table
            query_kmeans = f"""
            CREATE OR REPLACE TABLE `{dataset}.kmeans_results`
            CLUSTER BY CENTROID_ID, run_id AS
            SELECT
                *
            FROM
                ML.PREDICT(MODEL `{dataset}.kmeans_model`,
                    (SELECT
                        run_id,
                        ARRAY(
                            SELECT distance.{grouping_method} 
                            FROM UNNEST(distances) AS distance 
                            WHERE distance.run_id_b <= {num_features}  
                        ) AS features
                    FROM
                        `{dataset}.distance_matrix`)
                ) AS predictions
            """
            s = time.time()
            client.query(query_kmeans).result()  # Execute the model creation
            print(f"K-means clustering results saved successfully in {round(time.time() - s, 3)} seconds.")

    fixed_time_quantiles = f"""
    WITH daily_data AS (
        SELECT 
            date, 
            value,
            run_id,
            ROW_NUMBER() OVER (PARTITION BY date ORDER BY value) AS row_num,
            COUNT(*) OVER (PARTITION BY date) AS total_rows
        FROM `{dataset}.data`
    ),

    -- Joining with kmeans_results to attach CENTROID_ID
    centroid_data AS (
        SELECT 
            d.date,
            d.value,
            d.run_id,
            k.CENTROID_ID
        FROM daily_data d
        JOIN `{f'{dataset}.kmeans_results' if kmeans_table is False else kmeans_table}` k ON d.run_id = k.run_id -- Adjust the join condition if necessary
    )

    SELECT 
        CENTROID_ID,
        date,
        PERCENTILE_CONT(value, 0) OVER (PARTITION BY CENTROID_ID, date) AS Min,
        PERCENTILE_CONT(value, {(1-confidence)/2}) OVER (PARTITION BY CENTROID_ID, date) AS LowBound,
        PERCENTILE_CONT(value, 0.25) OVER (PARTITION BY CENTROID_ID, date) AS Q1,
        PERCENTILE_CONT(value, 0.50) OVER (PARTITION BY CENTROID_ID, date) AS Median,
        PERCENTILE_CONT(value, 0.75) OVER (PARTITION BY CENTROID_ID, date) AS Q3,
        PERCENTILE_CONT(value, {(1+confidence)/2}) OVER (PARTITION BY CENTROID_ID, date) AS HighBound,
        PERCENTILE_CONT(value, 1) OVER (PARTITION BY CENTROID_ID, date) AS Max
    FROM centroid_data
    GROUP BY CENTROID_ID, date, value
    ORDER BY CENTROID_ID, date;
    """
    plt_ftq = client.query(fixed_time_quantiles).result().to_dataframe()  # Execute the query to create the table
    print("Data pulled successfully.")

    # a monstrosity of a query
    get_outlying_points = f"""
    WITH daily_data AS (
        SELECT 
            date, 
            value,
            run_id
        FROM `{dataset}.data`  
    ),

    centroid_data AS (
        SELECT 
            d.date,
            d.value,
            d.run_id,
            k.CENTROID_ID
        FROM daily_data d
        JOIN `{dataset}.kmeans_results` k ON d.run_id = k.run_id
    ),

    percentile_data AS (
        SELECT 
            CENTROID_ID,
            date,
            PERCENTILE_CONT(value, {(1-confidence)/2}) OVER (PARTITION BY CENTROID_ID, date) AS LowBound,
            PERCENTILE_CONT(value, {(1+confidence)/2}) OVER (PARTITION BY CENTROID_ID, date) AS HighBound
        FROM centroid_data
        GROUP BY CENTROID_ID, date, value
    )

    -- Main query to filter points outside the 90% interval
    SELECT DISTINCT
        cd.CENTROID_ID,
        cd.date,
        cd.value
    FROM centroid_data cd
    JOIN percentile_data pd
    ON cd.CENTROID_ID = pd.CENTROID_ID
    AND cd.date = pd.date
    WHERE cd.value < pd.LowBound
    OR cd.value > pd.HighBound
    ORDER BY cd.CENTROID_ID, cd.date;

    """
    plt_outlying_points = client.query(get_outlying_points).result().to_dataframe()  # Execute the query to create the table
    print("Data pulled successfully.")

    if delete_data:
        client.delete_dataset(
            dataset,
            delete_contents=True,  # Set to False if you only want to delete an empty dataset
            not_found_ok=True      # If True, no error is raised if the dataset does not exist
        )
        print(f"BigQuery dataset `{client.project}.{dataset}` removed successfully, or it did not exist.")

    # Create graph

    colors = netsi.layout.colorway

    fig = go.Figure()
    fig.update_layout(
        title={
            'text': f"Traditional Boxplot",
        },
        xaxis_title="Date",
        yaxis_title="Susceptibility",
        template=netsi
        
    )
    # fig.update_xaxes(range=[pd.Timestamp("2009-09-01"), pd.Timestamp("2010-02-17")])
    # fig.update_yaxes(range=[0, 35000])

    try:
        plt_ftq.set_index('date', inplace=True)
    except Exception:
        pass

    for group in plt_ftq['CENTROID_ID'].unique():
        df_group = plt_ftq[plt_ftq['CENTROID_ID'] == group]

        if full_range:
            # FULL RANGE
            fig.add_trace(go.Scatter(
                name=f'Minimum',
                x=df_group.index,
                y=df_group['Min'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                showlegend=False,
                legendgroup=str(group)  # Assign to legend group
            ))
            fig.add_trace(go.Scatter(
                name=f'Full Range',
                x=df_group.index,
                y=df_group['Max'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                fillcolor=hex_to_rgba(colors[group-1], .2),
                fill='tonexty',
                showlegend=False,
                legendgroup=str(group)  # Assign to legend group
            ))
        
        
        # MIDDLE 90%
        fig.add_trace(go.Scatter(
            name=f'Minimum',
            x=df_group.index,
            y=df_group['LowBound'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            showlegend=False,
            legendgroup=str(group)  # Assign to legend group
        ))
        fig.add_trace(go.Scatter(
            name=f'Middle {confidence * 100}%',
            x=df_group.index,
            y=df_group['HighBound'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            fillcolor=hex_to_rgba(colors[group-1], .2),
            fill='tonexty',
            showlegend=False,
            legendgroup=str(group)  # Assign to legend group
        ))
        
        # MIDDLE 50%
        fig.add_trace(go.Scatter(
            name=f'Minimum',
            x=df_group.index,
            y=df_group['Q1'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            showlegend=False,
            legendgroup=str(group)  # Assign to legend group
        ))
        fig.add_trace(go.Scatter(
            name=f'Group {group}',
            x=df_group.index,
            y=df_group['Q3'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            fillcolor=hex_to_rgba(colors[group-1], .3),
            fill='tonexty',
            showlegend=True,
            legendgroup=str(group)  # Assign to legend group
        ))
        
        
        fig.add_trace(go.Scatter(
            name=f'Median',
            x=df_group.index,
            y=df_group['Median'],
            marker=dict(color=hex_to_rgba(colors[group-1], 1)),
            line=dict(width=1),
            mode='lines',
            showlegend=False,
            legendgroup=str(group)
        ))
        
        if outlying_points:
            fig.add_trace(go.Scatter(
            name=f'Outlying Points',
            x=plt_outlying_points[plt_outlying_points['CENTROID_ID'] == group]['date'],
            y=plt_outlying_points[plt_outlying_points['CENTROID_ID'] == group]['value'],
            mode='markers',
            marker=dict(color=hex_to_rgba(colors[group-1], .1)),
            showlegend=False,
            legendgroup=str(group)  # Assign to legend group
        ))
        
    return fig