import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from tabulate import tabulate

from IPython.display import display, HTML

from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt


class EdaHelperFunctions:

    def __init__(self):
        pass


    def estimate_dataset_size(self ,  df: pd.DataFrame) -> str:
        """
        Estimates the memory usage of a DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame for which to calculate memory usage.

        Returns:
        - str: The estimated memory usage in a human-readable format (bytes, KB, MB, or GB).
        """
        # Calculate memory usage of DataFrame in bytes
        mem_usage_bytes = df.memory_usage(deep=True).sum()

        # Convert bytes to a more readable format
        if mem_usage_bytes < 1024:
            size_str = f"{mem_usage_bytes} bytes"
        elif mem_usage_bytes < 1024 ** 2:
            size_str = f"{mem_usage_bytes / 1024:.2f} KB"
        elif mem_usage_bytes < 1024 ** 3:
            size_str = f"{mem_usage_bytes / 1024 ** 2:.2f} MB"
        else:
            size_str = f"{mem_usage_bytes / 1024 ** 3:.2f} GB"

        print(f"Estimated dataset size: {size_str}")
        return size_str


    def display_df(self, df: pd.DataFrame):
        """
        Renders and displays a pandas DataFrame as a styled HTML table within a Jupyter notebook or Databricks environment.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame to be displayed. This function assumes the input is a valid pandas DataFrame.

        Raises:
        ------
        ValueError
            If the input is not a pandas DataFrame.

        Notes:
        -----
        - The table styling uses a minimalistic approach with subtle borders and simple font settings.
        - Numerical values are formatted to 10 decimal places, while non-numeric values are displayed as strings.
        - Missing values are represented as 'NaN'.
        """

        # Verify that the input is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Apply minimalistic styling to the DataFrame
        styled_df = df.style.set_table_styles(
            [
                # Style for table headers
                {'selector': 'th', 'props': [
                    ('background-color', '#f5f5f5'),  # Light grey background for headers
                    ('color', 'black'),  # Black text
                    ('text-align', 'center'),
                    ('border', '1px solid #dcdcdc'),
                    ('font-family', 'Arial, sans-serif'),
                    ('font-size', '12pt'),
                    ('padding', '8px')  # Padding within header cells
                ]},
                # Style for table cells
                {'selector': 'td', 'props': [
                    ('text-align', 'center'),
                    ('border', '1px solid #dcdcdc'),
                    ('font-family', 'Arial, sans-serif'),
                    ('padding', '8px'),  # Padding within cells
                    ('font-size', '10pt')
                ]}
            ]
        ).set_properties(
            **{
                'border': '1px solid #dcdcdc',  # Border for all table cells
                'padding': '5px',  # Padding within each cell
                'font-size': '10pt'  # Font size for table text
            }
        ).format(precision=10, na_rep='NaN', formatter={
            col: '{:.10g}'.format if df[col].dtype == 'float' else str for col in df.columns
        })

    def overview_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates an overview table summarizing key characteristics of the provided DataFrame.

        Parameters:
        ----------
        data : pd.DataFrame
            The DataFrame for which the summary information is to be generated.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the following summary statistics:
            - 'number_samples': Total number of rows in the dataset.
            - 'number_features': Total number of columns in the dataset.
            - 'n_numeric_features': Number of numerical features in the dataset.
            - 'n_categorical_features': Number of categorical features in the dataset.
            - 'dataset_total_missing_values': Total number of missing values in the dataset.
            - 'number_duplicates': Total number of duplicate rows in the dataset.
            - 'missing_values_per_feature': A Series with the count of missing values for each feature.

        Notes:
        -----
        - Numerical features are identified using the data types "number".
        - Categorical features are identified using the data types "object" and "category".
        """
        # Calculate the number of samples (rows) and features (columns)
        number_samples = data.shape[0]
        number_features = data.shape[1]

        # Calculate the number of numerical features
        number_num = len(data.select_dtypes(include=["number"]).columns)

        # Calculate the number of categorical features
        number_categorical = len(data.select_dtypes(include=["object", "category"]).columns)

        # Calculate the total number of duplicate rows
        number_duplicates = data.duplicated().sum()

        # Create a summary table
        table = {
            "number_samples": [number_samples],
            "number_features": [number_features],
            "n_numeric_features": [number_num],
            "n_categorical_features": [number_categorical],
            "dataset_total_missing_values": [data.isna().sum().sum()],
            "number_duplicates": [number_duplicates]
        }

        # Convert the summary table to a DataFrame
        table = pd.DataFrame(table)

        return table

    def observe_data(self, data: pd.DataFrame, n: int = 5, head: bool = True) -> pd.DataFrame:
        """
        Observes the top or bottom 'n' rows of a pandas DataFrame.

        Parameters:
        -----------
        data : pd.DataFrame
            The DataFrame to observe.
        n : int, optional
            Number of rows to display. Default is 5.
        head : bool, optional
            If True, displays the first 'n' rows; if False, displays the last 'n' rows. Default is True.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the observed rows.

        Examples:
        ---------
        # Display the first 5 rows of the DataFrame
        observe_data(data)

        # Display the last 3 rows of the DataFrame
        observe_data(data, n=3, head=False)
        """
        if head:
            return data.head(n)
        else:
            return data.tail(n)

    def bar_plot(self, x, y, title="Cool Bar Plot", xlabel="X-axis", ylabel="Y-axis", rotation=80, annotation=True,
                 annotation_as_int=True, round_decimals=3, fig_size=(22, 12), threshold=None, log_scale=True,
                 annotation_rotation=45):
        """
        Creates and displays a bar plot with cool styling using matplotlib and optionally applies a logarithmic scale
        to the y-axis.

        Parameters:
        ----------
        x : list or array-like
            The x-axis data (categories or labels).
        y : list or array-like
            The y-axis data (values corresponding to x).
        title : str, optional
            The title of the plot (default is "Cool Bar Plot").
        xlabel : str, optional
            The label for the x-axis (default is "X-axis").
        ylabel : str, optional
            The label for the y-axis (default is "Y-axis").
        rotation : int, optional
            The rotation angle for x-axis labels (default is 90 degrees).
        annotation : bool, optional
            If True, display annotations on the bars (default is False).
        annotation_as_int : bool, optional
            If True, annotations will be displayed as integers (default is False).
        round_decimals : int, optional
            The number of decimal places to round the annotations (default is 3).
        fig_size : tuple, optional
            The size of the figure in inches (default is (10, 5)).
        threshold : float, optional
            A threshold value to be displayed as a horizontal line on the plot (default is None).
        log_scale : bool, optional
            If True, applies a log scale to the y-axis (default is False).
        annotation_rotation : int, optional
            The rotation angle for annotations (default is 45 degrees).
        """
        # Create a figure and axis with a custom size
        fig, ax = plt.subplots(figsize=fig_size)

        # Create a bar plot
        bars = ax.bar(x, y, color='#1f77b4', edgecolor='black')

        # Add labels, title, and grid with cool styling
        ax.set_title(title, fontsize=20, fontweight='bold', color='#333333')
        ax.set_xlabel(xlabel, fontsize=16, fontweight='bold', color='#555555', fontfamily='sans-serif')
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold', color='#555555', fontfamily='sans-serif')

        # Customize the grid and background
        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7, color='#aaaaaa')
        ax.set_facecolor('#f0f0f0')

        # Rotate x-axis labels
        plt.xticks(rotation=rotation, fontsize=12, fontweight='bold', fontfamily='sans-serif')

        # Apply logarithmic scaling to the y-axis if log_scale is True
        if log_scale:
            ax.set_yscale('log')

        # Add annotations if the parameter is set to True
        if annotation:
            for bar in bars:
                height = bar.get_height()

                # Format the annotation based on the annotation_as_int parameter and round_decimals
                if annotation_as_int:
                    label = f'{int(height)}'
                else:
                    label = f'{round(height, round_decimals)}'

                ax.annotate(label,  # Format the annotation with or without decimal places
                            xy=(bar.get_x() + bar.get_width() / 2, height),  # Annotation position
                            xytext=(0, 3),  # Offset the text slightly above the bar
                            textcoords="offset points",
                            ha='center', va='bottom',  # Align horizontally to the center and vertically to the bottom
                            fontsize=12, fontweight='bold', color='#333333',  # Style the annotation
                            rotation=annotation_rotation)  # Add the rotation for annotation

        # Plot the threshold line if the parameter is provided
        if threshold is not None:
            ax.axhline(y=threshold, color='green', linestyle='--', linewidth=3, label=f'Threshold: {threshold}')
            ax.legend()

        # Display the plot
        plt.tight_layout()
        return fig

    # Function to generate a summary DataFrame
    def info_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary DataFrame for the given DataFrame.

        Parameters:
        data (pd.DataFrame): The DataFrame for which to generate the summary.

        Returns:
        pd.DataFrame: A summary DataFrame containing the column names, non-null counts, and data types.
        """
        # Create the summary DataFrame
        summary_df = pd.DataFrame({
            'features_name': data.columns,
            'Non-Null Count': data.notnull().sum(),
            'Dtype': data.dtypes
        })

        return summary_df

    def numeric_and_categorical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies and separates numerical, categorical, and timestamp columns within a given DataFrame.

        This function analyzes the data types of the columns in the input DataFrame and categorizes them into three distinct groups: numerical features, categorical features, and timestamp features. The function then returns a DataFrame containing these groupings.

        Parameters:
        -----------
        data : pd.DataFrame
            The input DataFrame whose columns are to be analyzed and categorized.

        Returns:
        --------
        pd.DataFrame
            A DataFrame with the following three columns:
            - 'Numerical Features': Lists the names of columns that contain numerical data.
            - 'Categorical Features': Lists the names of columns that contain categorical data.
            - 'Timestamp Features': Lists the names of columns that contain datetime data.

        Example:
        --------
        # Extract a DataFrame of numerical, categorical, and timestamp columns from the input DataFrame
        features_df = numeric_and_categorical_columns(data)

        Notes:
        ------
        - Numerical features are identified by their 'number' data type (e.g., int, float).
        - Categorical features include columns of 'object' or 'category' data types.
        - Timestamp features are identified by the 'datetime' data type.
        """
        # Extract columns with numerical data types
        numerical_features = data.select_dtypes(include=["number"]).columns.tolist()

        # Extract columns with categorical data types
        categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Extract columns with timestamp (datetime) data types
        timestamp_features = data.select_dtypes(include=['datetime']).columns.tolist()

        # Combine the identified columns into a single DataFrame
        features_df = pd.DataFrame({
            "Numerical Features": pd.Series(numerical_features),
            "Categorical Features": pd.Series(categorical_features),
            "Timestamp Features": pd.Series(timestamp_features)
        })

        return features_df

    def show_column_unique_counts(self, data: pd.DataFrame, decimal_places: int = 2) -> pd.DataFrame:
        """
        Calculate and display the count of unique values and their ratio in each column of a DataFrame.

        Parameters:
        -----------
        data : pd.DataFrame
            The DataFrame for which unique value counts are to be calculated.
        decimal_places : int, optional
            Number of decimal places to round the ratios to (default is 2).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the count of unique values and their ratio for each column.
            The index of the DataFrame represents column names, and columns are:
            - 'Feature': The name of the column (feature) in the DataFrame
            - 'Unique Values': Number of unique values in each column
            - 'Ratio': Percentage of unique values relative to the total number of rows in each column, rounded to the specified decimal places.

        Examples:
        ---------
        # Display unique value counts and ratios for each column in a DataFrame
        show_column_unique_counts(data)
        """
        # Calculate unique values and their ratios
        unique_values = pd.DataFrame(data.nunique(), columns=["Unique Values"])
        ratio = pd.DataFrame((data.nunique() / data.shape[0]) * 100, columns=["Ratio"])

        # Round the ratios to the specified number of decimal places
        ratio["Ratio"] = ratio["Ratio"].round(decimal_places)

        # Combine the unique values and ratios into one DataFrame
        table = pd.concat([unique_values, ratio], axis=1)

        # Add the feature (column name) as a new column
        table['Feature'] = table.index

        # Reset the index so that 'Feature' becomes a column
        table = table.reset_index(drop=True)

        # Reorder the columns to place 'Feature' first
        table = table[['Feature', 'Unique Values', 'Ratio']]

        return table

    def show_column_unique_counts(self, data: pd.DataFrame, sort_by: str = "Unique Values",
                                  ascending: bool = True) -> pd.DataFrame:
        """
        Calculate and display the count of unique values in each categorical column of a DataFrame.

        Parameters:
        -----------
        data : pd.DataFrame
            The DataFrame for which unique value counts are to be calculated.
        sort_by : str, optional
            The column name to sort the result by ('Feature' or 'Unique Values').
            Default is 'Unique Values'.
        ascending : bool, optional
            If True, sort in ascending order; if False, sort in descending order.
            Default is True (ascending).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the count of unique values for each categorical column.
            The index of the DataFrame represents column names, and columns are:
            - 'Feature': The name of the column (feature) in the DataFrame.
            - 'Unique Values': Number of unique values in each categorical column.

        Examples:
        ---------
        # Display unique value counts for each categorical column in a DataFrame
        show_column_unique_counts(data)

        # Display sorted by unique values in descending order
        show_column_unique_counts(data, sort_by='Unique Values', ascending=False)
        """
        # Filter to only categorical columns
        categorical_data = data.select_dtypes(include=['object', 'category'])

        # Calculate unique values for categorical columns
        unique_values = pd.DataFrame(categorical_data.nunique(), columns=["Unique Values"])

        # Add the feature (column name) as a new column
        unique_values['Feature'] = unique_values.index

        # Reset the index so that 'Feature' becomes a column
        unique_values = unique_values.reset_index(drop=True)

        # Reorder the columns to place 'Feature' first
        unique_values = unique_values[['Feature', 'Unique Values']]

        # Sort the table if sort_by is provided
        if sort_by in ['Feature', 'Unique Values']:
            unique_values = unique_values.sort_values(by=sort_by, ascending=ascending)

        return unique_values

    def show_column_missing_counts(self, data: pd.DataFrame, decimal_places: int = 2, sort_by: str = "Missing Values",
                                   ascending: bool = True) -> pd.DataFrame:
        """
        Calculate and display the count and ratio of missing values (NaN) in each column of a DataFrame.

        Parameters:
        -----------
        data : pd.DataFrame
            The DataFrame for which missing value counts are to be calculated.
        decimal_places : int, optional
            Number of decimal places to round the ratios to (default is 2).
        sort_by : str, optional
            The column name to sort the result by ('Feature', 'Missing Values', or 'Ratio').
            Default is None, which means no sorting.
        ascending : bool, optional
            If True, sort in ascending order; if False, sort in descending order.
            Default is True (ascending).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the count and ratio of missing values for each column.
            The index of the DataFrame represents column names, and columns are:
            - 'Feature': The name of the column (feature) in the DataFrame
            - 'Missing Values': Number of missing values in each column
            - 'Ratio': Percentage of missing values relative to the total number of rows in each column, rounded to the specified decimal places.

        Examples:
        ---------
        # Display missing value counts and ratios for each column in a DataFrame
        show_column_missing_counts(data)

        # Display sorted by missing values in descending order
        show_column_missing_counts(data, sort_by='Missing Values', ascending=False)
        """
        # Calculate missing values and their ratios
        missing_values = pd.DataFrame(data.isnull().sum(), columns=["Missing Values"])
        ratio = pd.DataFrame((data.isnull().sum() / data.shape[0]) * 100, columns=["Ratio"])

        # Round the ratios to the specified number of decimal places
        ratio["Ratio"] = ratio["Ratio"].round(decimal_places)

        # Combine the missing values and ratios into one DataFrame
        table = pd.concat([missing_values, ratio], axis=1)

        # Add the feature (column name) as a new column
        table['Feature'] = table.index

        # Reset the index so that 'Feature' becomes a column
        table = table.reset_index(drop=True)

        # Reorder the columns to place 'Feature' first
        table = table[['Feature', 'Missing Values', 'Ratio']]

        # Sort the table if sort_by is provided
        if sort_by in ['Feature', 'Missing Values', 'Ratio']:
            table = table.sort_values(by=sort_by, ascending=ascending)

        return table

    def show_zeros_table(self, data: pd.DataFrame, threshold: float = 0.1, decimal_places: int = 2) -> pd.DataFrame:
        """
        Generate a table displaying columns with zero values exceeding a specified threshold.

        Parameters:
        -----------
        data : pd.DataFrame
            The DataFrame to analyze for zero values.
        threshold : float, optional
            The threshold above which the ratio of zeros in a column is considered significant. Default is 0.1.
        decimal_places : int, optional
            Number of decimal places to round the ratio of zeros to (default is 2).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing columns with zero values exceeding the specified threshold.
            The DataFrame includes columns:
            - 'feature': Name of the column
            - 'n_zeros': Number of zero values in the column
            - 'ratio': Percentage of zero values relative to the total number of rows, rounded to the specified decimal places.

        Examples:
        ---------
        # Display columns with zero values exceeding a threshold of 10% and round the ratio to 3 decimal places
        show_zeros_table(data, threshold=0.1, decimal_places=3)
        """
        # Identify columns with zero values exceeding the threshold
        zero_cols = [col for col in data if data[col].dtype in ['int64', 'float64'] and
                     (data.loc[data[col] == 0, col].count() / data.shape[0]) > threshold]

        zero_data = []

        for col in zero_cols:
            n_zeros = data.loc[data[col] == 0, col].count()
            ratio = (n_zeros / data.shape[0]) * 100
            zero_data.append({
                "feature": col,
                "n_zeros": n_zeros,
                "ratio": round(ratio, decimal_places)
            })

        zero_df = pd.DataFrame(zero_data)

        return zero_df

    def calculate_statistics(self, data: pd.DataFrame, low_quantile: float = 0.25, high_quantile: float = 0.75,
                             sort_by: str = "Mean", ascending: bool = True) -> pd.DataFrame:
        """
        Calculates outliers and various statistical metrics for each numerical column.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataframe.
        low_quantile : float, optional
            The low quantile value for IQR calculation. Default is 0.25.
        high_quantile : float, optional
            The high quantile value for IQR calculation. Default is 0.75.
        sort_by : str, optional
            The column name to sort the result by ('Feature', 'Outlier Count', 'Min', 'Max', 'Median', 'Mean', 'Standard Deviation', '1st Quantile', '25th Quantile', '75th Quantile', '95th Quantile', '99th Quantile').
            Default is None, which means no sorting.
        ascending : bool, optional
            If True, sort in ascending order; if False, sort in descending order.
            Default is True (ascending).

        Returns:
        --------
        pd.DataFrame
            A DataFrame with column names as index and statistical metrics as columns.

        Examples:
        ---------
        # Calculate and display statistics without sorting
        calculate_statistics(data)

        # Calculate and display statistics sorted by 'Mean' in descending order
        calculate_statistics(data, sort_by='Mean', ascending=False)
        """
        stats = {}
        numeric_data = data.select_dtypes(include=['number'])
        for column in numeric_data.columns:
            Q1 = data[column].quantile(low_quantile)
            Q3 = data[column].quantile(high_quantile)
            Q4 = data[column].quantile(0.95)
            Q01 = data[column].quantile(0.01)
            Q99 = data[column].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # If the lower_bound and upper_bound are the same, set outlier count to 0
            if lower_bound == upper_bound:
                outlier_count = 0
            else:
                outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
                outlier_count = outliers[column].count()

            # Calculating statistical metrics
            stats[column] = {
                'Feature': column,
                'Outlier Count': outlier_count,
                'Min': data[column].min(),
                'Max': data[column].max(),
                'Median': data[column].median(),
                'Mean': data[column].mean(),
                'Standard Deviation': data[column].std(),
                '1st Quantile': Q01,
                '25th Quantile': Q1,
                '75th Quantile': Q3,
                '95th Quantile': Q4,
                '99th Quantile': Q99
            }

        # Convert the dictionary into a DataFrame
        stats_df = pd.DataFrame.from_dict(stats, orient='index')

        # Sort the DataFrame if sort_by is provided
        if sort_by in stats_df.columns:
            stats_df = stats_df.sort_values(by=sort_by, ascending=ascending)

        return stats_df

    def plot_boxplots(self, dataframe):
        """
        Plots box plots for all numerical columns in the dataframe to visualize outliers.

        Parameters:
        dataframe (pd.DataFrame): The input dataframe containing the data.

        Returns:
        matplotlib.figure.Figure: The figure object containing the box plots.
        """
        # Select only numerical columns
        numeric_columns = dataframe.select_dtypes(include=['number']).columns

        # Define the number of subplots needed
        num_columns = len(numeric_columns)
        num_rows = (num_columns + 2) // 3  # 3 columns per row

        # Create a figure for the plots
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

        # Flatten axes for easier iteration, especially if there are fewer plots than subplots
        axes = axes.flatten()

        for i, column in enumerate(numeric_columns):
            sns.boxplot(y=dataframe[column], ax=axes[i])
            axes[i].set_title(f'Boxplot of {column}')

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()  # Adjust subplots to fit into figure area.

        return fig  # Return the figure object

    def variance_analysis(self, df, features=None, annotate=False, annotation_rotation: int = 80, log_scale: bool = False,
                          fig_size: tuple = (20, 12), title: str = "Variance Analysis for Numerical Features"):
        """
        Analyzes the variance of specified numerical features in the given DataFrame.
        If no features are specified, analyzes all numerical features.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing numerical features.
        features (list of str, optional): A list of the names of the numerical features to analyze. If None, analyzes all numerical features.
        annotate (bool): Whether to add annotations to the plot.

        Returns:
        pd.DataFrame: A DataFrame with columns 'Feature' and 'Variance' showing the variance of each numerical feature, or just the specified features.
        plt.Figure: The figure object for the plot.
        """

        # Ensure that the DataFrame contains numerical features only
        numerical_features = df.select_dtypes(include=[np.number])

        if features:
            # Validate that all specified features are in the DataFrame
            missing_features = [feature for feature in features if feature not in numerical_features.columns]
            if missing_features:
                raise ValueError(f"Features {missing_features} not found in the DataFrame.")
            # Calculate variance for the specified features only
            variance_series = numerical_features[features].var()
            variance_df = pd.DataFrame({
                'Feature': variance_series.index,
                'Variance': variance_series.values
            })
        else:
            # Calculate variance for each numerical feature
            variance_series = numerical_features.var()

            # Create a DataFrame to store the variance of each feature
            variance_df = pd.DataFrame({
                'Feature': variance_series.index,
                'Variance': variance_series.values
            })

            # Sort the DataFrame by variance in descending order for better readability
            variance_df = variance_df.sort_values(by='Variance', ascending=False).reset_index(drop=True)

        # Print DataFrame
        self.display_df(variance_df)

        # Use the bar_plot function to create the plot with customized styling
        fig = self.bar_plot(

            x=variance_df['Feature'],
            y=variance_df['Variance'],
            title=title,
            xlabel='Feature',
            ylabel='Variance',
            rotation=90,
            annotation=annotate,
            fig_size=(20, 12),
            log_scale=log_scale,  # Set to True if variance values span several orders of magnitude
            annotation_rotation=annotation_rotation,
            annotation_as_int=False

        )

        return variance_df, fig

    def plot_histograms(self, df: pd.DataFrame, columns: list[str] = None, n_cols: int = 3, annotate: bool = False,
                        type: str = "hist") -> plt.Figure:
        """
        Plots histograms or box plots for specified columns in a DataFrame.

        Parameters:
        - df (DataFrame): The DataFrame containing the data to be plotted.
        - columns (list[str], optional): List of column names to plot. Defaults to None, plotting all numerical columns.
        - n_cols (int): Number of columns in the subplot grid. Defaults to 1.
        - annotate (bool): Whether to annotate the plots with values. Defaults to False.
        - type (str): Type of plot to create ('hist' for histograms or 'box' for box plots). Defaults to 'hist'.

        Raises:
        - KeyError: If an unsupported plot type is specified.

        Returns:
        - plt.Figure: The matplotlib Figure object containing the plots.
        """

        if columns is None:
            # Filter the DataFrame to include only numerical columns if columns parameter is not specified
            data = df.select_dtypes(include=["number"])
        else:
            data = df[columns]

        # Calculate the number of rows needed for the subplots
        n_rows = (data.shape[1] + n_cols - 1) // n_cols  # ceiling division

        # Create the figure and subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))

        # If only one subplot is created, `axes` is not an array
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])

        # Flatten the axes array for easier iteration
        axes_flat = axes.flatten()

        # Iterate over the numerical columns and plot histograms
        for i, col in enumerate(data.columns):
            ax = axes_flat[i]
            if type == "hist":
                sns.histplot(data[col], ax=ax, bins='auto', kde=True)
            elif type == "box":
                sns.boxplot(x=data[col], ax=ax, orient="h")
            else:
                raise KeyError("Please provide a valid type: 'box' or 'hist'")

            ax.set_title(f'Column: {col}')

            # Rotate x-axis labels
            ax.tick_params(axis='x', labelrotation=45)

            if annotate and type == "hist":
                # Annotate histogram bars with counts
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                textcoords='offset points')

        # Remove any unused subplots
        for i in range(data.shape[1], len(axes_flat)):
            fig.delaxes(axes_flat[i])

        # Adjust layout for better spacing and display the plot
        plt.tight_layout()
        #plt.show()

    def analyze_categorical_features(self, df: pd.DataFrame, threshold: int = 30) -> dict:
        """
        Analyze categorical features in a DataFrame.

        This function processes the categorical columns of the given DataFrame,
        generating a dictionary of DataFrames. Each DataFrame contains the count
        and ratio of each unique value for a categorical feature, excluding those
        with more than the specified threshold of unique values. Additionally,
        returns a DataFrame with features that were skipped due to exceeding
        the unique values threshold.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing the data to analyze.
        threshold (int): The maximum number of unique values allowed for a feature to be included.

        Returns:
        dict: A dictionary where keys are column names of categorical features and
              values are DataFrames with 'Feature', 'Subgroup', 'Count', and 'Ratio' of unique values.
        DataFrame: A DataFrame containing the names of features that were skipped,
                   along with an alert message.
        """

        # Dictionary to store DataFrames for each categorical feature
        categorical_dfs = {}

        # List to keep track of skipped features
        skipped_features = []

        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns

        # Iterate over each categorical column
        for column in categorical_columns:
            # Check the number of unique values in the column
            unique_values_count = df[column].nunique()

            # Skip features with more than the specified threshold of unique values
            if unique_values_count > threshold:
                # Add skipped feature information to the list
                skipped_features.append({
                    'Feature': column,
                    'Alert': f"Feature '{column}' was skipped because it has {unique_values_count} unique values, exceeding the threshold of {threshold}."
                })
                continue

            # Get the counts of each unique value in the column
            value_counts = df[column].value_counts()

            # Calculate the ratio of each unique value
            total_count = len(df[column])
            value_ratios = (value_counts / total_count) * 100

            # Create a DataFrame for the current feature
            feature_df = pd.DataFrame({
                'Feature': column,
                'Subgroup': value_counts.index,
                'Count': value_counts.values,
                'Ratio': value_ratios.values.round(3)
            })

            # Add the DataFrame to the dictionary
            categorical_dfs[column] = feature_df

        # Convert skipped features list to DataFrame
        skipped_features_df = pd.DataFrame(skipped_features)

        return categorical_dfs, skipped_features_df

    def calculate_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the correlation matrix for numerical columns in a DataFrame and apply default display settings.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

        Returns:
        pd.DataFrame: The correlation matrix.
        """
        # Select numerical columns
        numerical_data = df.select_dtypes(include=["number"])

        # Calculate correlation matrix
        correlation_matrix = numerical_data.corr()

        return correlation_matrix

    def plot_correlation_heatmap(self, correlation_matrix, title='Correlation Heatmap', cmap='coolwarm',
                                 annot=True) -> plt.Figure:
        """
        Generates a heatmap from a correlation matrix.

        Parameters:
        correlation_matrix (pd.DataFrame): A Pandas DataFrame containing the correlation matrix.
        title (str): Title of the heatmap plot. Default is 'Correlation Heatmap'.
        cmap (str): Colormap to use for the heatmap. Default is 'coolwarm'.
        annot (bool): Whether to annotate the heatmap with correlation coefficients. Default is True.

        Returns:
        plt.Figure: The matplotlib Figure object containing the heatmap plot.
        """

        # Check if the input is a Pandas DataFrame
        if not isinstance(correlation_matrix, pd.DataFrame):
            raise TypeError("The correlation matrix must be a Pandas DataFrame.")

        # Verify that the DataFrame is square
        if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
            raise ValueError("The correlation matrix must be square.")

        # Create the heatmap
        fig = plt.figure(figsize=(20, 10))  # Set the figure size and capture the figure object
        sns.heatmap(correlation_matrix,
                    annot=annot,
                    cmap=cmap,
                    fmt='.2f',  # Format for annotations
                    linewidths=0.5,  # Width of the lines that will divide each cell
                    linecolor='black')  # Color of the lines dividing cells

        # Set plot title and labels
        plt.title(title, size=15)
        plt.xlabel('Variables')
        plt.ylabel('Variables')
        plt.tight_layout()  # Adjust layout to prevent overlap

    def cramers_v(self, x, y):

        """
        Computes Cramér's V statistic for association between two categorical variables.

        Parameters:
        - x (pd.Series): The first categorical variable.
        - y (pd.Series): The second categorical variable.

        Returns:
        - float: Cramér's V statistic.
        """
        crosstab = pd.crosstab(x, y)
        chi2_stat, p_value, dof, expected = chi2_contingency(crosstab, correction=False)
        n = crosstab.sum().sum()
        return np.sqrt(chi2_stat / (n * min(crosstab.shape) - 1))

    def categorical_correlation_matrix(self, data: pd.DataFrame) -> pd.DataFrame:

        """
        Creates a correlation matrix for categorical features in the DataFrame `data` using Cramér's V.

        Parameters:
        - data (pd.DataFrame): The input DataFrame containing the categorical data.

        Returns:
        - pd.DataFrame: A DataFrame containing Cramér's V statistics for each pair of categorical features.
        """
        # Select categorical columns
        categorical_features = data.select_dtypes(include=['object', 'category'])

        # Initialize an empty DataFrame for the correlation matrix
        corr_matrix = pd.DataFrame(index=categorical_features.columns, columns=categorical_features.columns)

        # Compute Cramér's V for each pair of categorical features
        for i in range(len(categorical_features.columns)):
            for j in range(i, len(categorical_features.columns)):
                feature_i = categorical_features.columns[i]
                feature_j = categorical_features.columns[j]
                if feature_i == feature_j:
                    corr_matrix.loc[feature_i, feature_j] = np.nan  # Avoid self-correlation
                else:
                    corr_value = self.cramers_v(categorical_features[feature_i], categorical_features[feature_j])
                    corr_matrix.loc[feature_i, feature_j] = corr_value
                    corr_matrix.loc[feature_j, feature_i] = corr_value  # Symmetric matrix

        # Convert all values to numeric, coercing errors to NaNs, then fill NaNs with 0
        corr_matrix = corr_matrix.apply(pd.to_numeric, errors='coerce')
        corr_matrix.fillna(0, inplace=True)

        return corr_matrix


    def plot_heatmap(self, matrix: pd.DataFrame, title: str = 'Categorical Correlation Heatmap') -> plt.Figure:
        """
        Plots a heatmap of the correlation matrix.

        Parameters:
        - matrix (pd.DataFrame): The correlation matrix to plot.
        - title (str): The title of the heatmap.

        Returns:
        - plt.Figure: The matplotlib Figure object containing the heatmap plot.
        """
        fig = plt.figure(figsize=(20, 10))  # Assign the figure to a variable
        sns.heatmap(matrix, annot=True, cmap='coolwarm', center=0, vmin=0, vmax=1, fmt='.2f', linewidths=0.5)
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()


    def create_line_plot(self, x, y, title, xlabel, ylabel):
        """
        Create a styled and professional line plot.

        Parameters:
        x (list or pd.Series): Data for the x-axis.
        y (list or pd.Series): Data for the y-axis.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        """
        # Set the style of seaborn
        sns.set(style="whitegrid")

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot the data
        sns.lineplot(x=x, y=y, marker='o', color='royalblue', linewidth=2.5, markersize=8)

        # Customize the plot with title and labels
        plt.title(title, fontsize=16, weight='bold', color='darkblue')
        plt.xlabel(xlabel, fontsize=14, weight='bold', color='darkblue')
        plt.ylabel(ylabel, fontsize=14, weight='bold', color='darkblue')

        # Improve the aesthetics
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Show the plot
        plt.tight_layout()
        plt.show()

    def process_grouped_dataframe(self, grouped_df):
        """
        Processes a DataFrame with MultiIndex columns by flattening the column headers
        and resetting the index.

        Parameters:
        grouped_df (pd.DataFrame): A DataFrame with MultiIndex columns and indices.

        Returns:
        pd.DataFrame: A DataFrame with flattened columns and a reset index.
        """
        # Flatten the MultiIndex columns by joining levels with an underscore
        grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]

        # Reset the index to convert the MultiIndex into standard columns
        grouped_df = grouped_df.reset_index()

        return grouped_df
