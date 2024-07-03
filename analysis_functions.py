from collections import Counter
from functools import reduce
from typing import List, Tuple, Union
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import t, wilcoxon
from scipy.stats import ttest_1samp
import scipy.stats as stats
from xml.etree.ElementTree import fromstring, ElementTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



def read_clean_csv_data(fileName : str  ,index_col : Union[int, str] =0) -> pd.DataFrame :
        """Function to load data set from a .csv file.

        Args:
            fileName (str, optional): The name of the .csv file.
            index_col (Union[int, str], optional): The index column of the resulting dataframe. Defaults to 0.

        Returns:
            pd.DataFrame: The cleaned and preprocessed dataframe.
        """ 

        ## Read the .csv file with the pandas read_csv method
        df = pd.read_csv( fileName ,index_col= index_col)
    
        ## Remove rows with missing values, accounting for mising values coded as '?'
        cols= df.columns
        for column in cols:
            df.loc[df[column] == '?', column] = np.nan
        df.dropna(axis = 0, inplace = True)

    
        
        return df

def convert_k_m_to_numeric(value):
        if 'k' in value:
            return float(value.replace('k', '')) * 1000
        elif 'm' in value:
            return float(value.replace('m', '')) * 1000000
        else:
            return float(value)

def quartiles_counts(df_parts: list, column_name: str, value_name: str) -> pd.Series:
        """
        Computes the count of rows in each quartile and  with a specified value in a specified column.
    
        Args:
            df_parts (list of pandas.DataFrame): A list of dataframes to compute counts for.
            column_name (str): The name of the column to check for the specified value.
            value_name (str): The value to check for in the specified column.
        
        Returns:
        pandas.Series: A series of counts for each quartile, with quartile names as indices.
        """
        counts_series = pd.Series({}, dtype=int)
        for i, df_part in enumerate(df_parts):
            x = (df_part[column_name] == value_name).sum()
            counts_series[f"Quartile {i+1}"] = x
        return counts_series
def six_parts_counts(df_parts: list, column_name: str, value_name: str) -> pd.Series:
        """
        If lenght of data lit <3 Computes the count of rows in each quartile if 2< lenght <5  count  in each quartil if lenght > 4 ads top and bottom with quartiles
        with a specified value in a specified column.
    
        Args:
            df_parts (list of pandas.DataFrame): A list of dataframes to compute counts for.
            column_name (str): The name of the column to check for the specified value.
            value_name (str): The value to check for in the specified column.
        
        Returns:
        pandas.Series: A series of counts for each quartile and top bottom dataframes, with parts  names as indices.
        """
        counts_series = pd.Series({}, dtype=int)
        if len(df_parts)>2:
            for i, df_part in enumerate(df_parts):
                x = (df_part[column_name] == value_name).sum()
                if x>0 :
                    if i == 4:
                        counts_series["Top20"] = x
                    elif i == 5:
                        counts_series["Bottom21"] = x
                    else:
                        counts_series[f"Quartile {i+1}"] = x
        
        if len(df_parts) < 3:
            for i, df_part in enumerate(df_parts):
                x = (df_part[column_name] == value_name).sum()
                if x>0 :
                    if i == 0:
                        counts_series["Above mean"] = x
                    elif i == 1:
                        counts_series["Below mean"] = x
                    
        return counts_series





def labeledBarChart(counts: pd.Series, xlabel: str = 'Name', ylabel: str = 'Count', 
                            title: str = "Title", figsize: Tuple[float, float] = (10,10), rotation: int = 0) -> None:
        """Creates a labeled bar chart from a pandas Series.

        Args:
            counts (pd.Series): The pandas series with the data to be plotted.
            xlabel (str, optional): The x-axis label. Defaults to 'Name'.
            ylabel (str, optional): The y-axis label. Defaults to 'Count'.
            title (str, optional): The title of the plot. Defaults to "Title".
            figsize (Tuple[float, float], optional): The size of the figure. Defaults to (10,10).

        Returns:
            None: Displays the labeled bar chart.
        """

        fig = plt.figure(figsize = figsize)
        ax = fig.gca()
        plt.xticks(rotation = rotation)
        counts_bars = ax.bar(counts.index, counts.values)
        # Add count labels to the bars
        for i, count in enumerate(counts.values):
            ax.text(i, count+2, str(count), ha='center', va='bottom')
         # Add x-axis and y-axis labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
         # Show the plot
        plt.show()
    
    
def t_test_confidence_intervals(data: np.ndarray,column:str,con_lvl: float=0.95):
     data = data[column].values
     # Calculate the sample mean and standard deviation
     sample_mean = np.mean(data)
     sample_std = np.std(data)

     # Calculate the sample size
     sample_size = len(data)
     # Calculate the critical value (two-tailed t-test)
     critical_value = stats.t.ppf((1 + con_lvl) / 2, df=sample_size - 1)

     # Calculate the standard error
     standard_error = sample_std / np.sqrt(sample_size)

     # Calculate the margin of error
     margin_of_error = critical_value * standard_error

     # Calculate the confidence interval
     lower_bound = sample_mean - margin_of_error
     upper_bound = sample_mean + margin_of_error

     # Print the results
     print("Confidence Interval ({}%): [{:.3f}, {:.3f}]".format(con_lvl * 100, lower_bound, upper_bound))

def test_of_pop_proportion_bigger(data: np.ndarray,column:str, variable : float):
     # Count the total number of reviews
    data_lenght = len(data[column])

    # Count the number of reviews with ratings higher than 4.0
    higher_ratings_reviews = len(data[data[column] > variable])

    # Calculate the point estimate for the population proportion
    point_estimate = higher_ratings_reviews / data_lenght

    # Calculate the standard error
    standard_error = np.sqrt((point_estimate * (1 - point_estimate)) / data_lenght)

    # Calculate the margin of error for a 95% confidence level (Z-score of 1.96)
    margin_of_error = 1.96 * standard_error

    # Calculate the confidence interval
    lower_bound = point_estimate - margin_of_error
    upper_bound = point_estimate + margin_of_error

    # Print the results
    print("margin of error:", margin_of_error)
    print("Point Estimate:", point_estimate)
    print("95% Confidence Interval: [{:.4f}, {:.4f}]".format(lower_bound, upper_bound))         
def pieChart(count: pd.Series, title: str = 'Title' , figsize: Tuple[float, float] = (8,8)) -> None:
        """Creates a pie chart from a pandas Series.

        Args:
            count (pd.Series): The pandas series with the data to be plotted.
            title (str, optional): The title of the plot. Defaults to 'Title'.
            figsize (Tuple[float, float], optional): The size of the figure.

        Returns:
            None: Displays the pie chart.    
        """
        fig = plt.figure(figsize = figsize)
        ax = fig.gca()
        ax.pie(count.values, labels = count.index, autopct='%1.1f%%')

        # Add title
        ax.set_title(title)

        # Show the plot
        plt.show()


def confidence_intervals(data: np.ndarray,conf_lvl = 0.95):
    data = data.values
    mean = data.mean()
    std = data.std()
    n = len(data)
    conf_int = t.interval(
    0.95, df=n - 1, loc=mean, scale=std / np.sqrt(n))
    return conf_int

def conf_int_pop_mean(data: np.ndarray):
     # Calculate sample statistics
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    sample_size = len(data)

    # Set the desired confidence level
    confidence_level = 0.95

    # Calculate the critical value (t-distribution)
    critical_value = stats.t.ppf((1 + confidence_level) / 2, df=sample_size - 1)

    # Calculate the standard error
    standard_error = sample_std / np.sqrt(sample_size)

    # Calculate the margin of error
    margin_of_error = critical_value * standard_error

    # Calculate the confidence interval
    confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

    # Print the results
    print("Sample Mean:", sample_mean)
    print("Sample Standard Deviation:", sample_std)
    print("Sample Size:", sample_size)
    print("Confidence Interval:", confidence_interval)
     


def samp1_ttest(data, null=0.0, alpha=0.05):
    """
    Perform a one-sample t-test on the data.

    Parameters:
    - data (array-like): The data array on which to perform the t-test.
    - null (float): The null hypothesis value to test against (default: 0.0).
    - alpha (float): The significance level for calculating the critical value (default: 0.05).

    Returns:
    - t_statistic (float): The calculated t-statistic.
    - p_value (float): The calculated p-value.
    """
    t_statistic, p_value = stats.ttest_1samp(data, null)

    # Calculate the critical value at the given significance level
    critical_value = stats.t.ppf(1 - alpha, len(data) - 1)

    # Compare the t-statistic with the critical value
    if t_statistic > critical_value:
        hypothesis_result = "Reject the null hypothesis"
    else:
        hypothesis_result = "Fail to reject the null hypothesis"

    # Print the results
    print("t-statistic:", t_statistic)
    print("p-value:", p_value)

    # Print the results
    print(f"One-sample t-test - Statistical Significance (p-value): {p_value:.4f}")
        
def wilcoxon_significance_and_intervals(data: np.ndarray) -> tuple:
    """
    Perform Wilcoxon signed-rank test and calculate confidence intervals.

    Args:
        data (np.ndarray): Array of paired/matched samples.

    Returns:
        tuple: Statistical significance (p-value) and confidence intervals.

    """

    # Perform the Wilcoxon signed-rank test
    statistic, p_value = wilcoxon(data)

    # Set the desired confidence level
    confidence_level = 0.95

    # Calculate the confidence intervals
    n = len(data)
    z_critical = 1.96  # For a 95% confidence level (two-tailed test)

    mean = np.mean(data)
    std_dev = np.std(data)

    margin_of_error = z_critical * (std_dev / np.sqrt(n))

    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    print(f"Statistical Significance (p-value): {p_value:.4f}")
    print(f"Confidence Interval: [{lower_bound:.2f}, {upper_bound:.2f}]")
    # Return the statistical significance and confidence intervals
    return p_value, (lower_bound, upper_bound)
    # Print the statistical significance and confidence intervals
    print(f"Statistical Significance (p-value): {p_value:.4f}")
    print(f"Confidence Interval: [{lower_bound:.2f}, {upper_bound:.2f}]")
def population_mean(data: np.ndarray)      :
     data = np.array(data)
     # Calculate the population mean
     population_mean = data.mean()

     # Calculate the confidence interval
     confidence_level = 0.95
     alpha = 1 - confidence_level

     z_critical = stats.norm.ppf(1 - alpha / 2)  # Z-value for 95% confidence interval

     standard_error = data.std() / np.sqrt(len(data))
     margin_of_error = z_critical * standard_error

     lower_bound = population_mean - margin_of_error
     upper_bound = population_mean + margin_of_error

     # Print the results
     print("Population Mean:", population_mean)
     print("Confidence Interval:", (lower_bound, upper_bound))   


def varible_mean_Zhypothesis(data: np.ndarray,alpha = 0.05,null_mean = 4.62):
     data = np.array(data)

     # Calculate sample statistics
     sample_mean = np.mean(data)
     sample_std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
     sample_size = len(data)

     # Calculate the test statistic (z-score)
     z_score = (sample_mean - null_mean) / (sample_std / np.sqrt(sample_size))

     # Calculate the critical value (z-value) for two-tailed test
     critical_value = stats.norm.ppf(1 - alpha / 2)

     # Calculate the p-value
     p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

     # Compare the test statistic with critical value and p-value with alpha
     if abs(z_score) > critical_value:
      print("Reject the null hypothesis")
     else:
      print("Fail to reject the null hypothesis")

     print("Sample Mean:", sample_mean)
     print("Sample Standard Deviation:", sample_std)
     print("Sample Size:", sample_size)
     print("Test Statistic (z-score):", z_score)
     print("Critical Value (z-value):", critical_value)
     print("P-value:", p_value)
     


def bootstrap_confidence_interval(data: np.ndarray, num_bootstrap_samples: int=1000, confidence_level: float=0.95) -> tuple:
    """
    Calculate the confidence interval using bootstrapping.

    Args:
        data (array-like): The original data.
        num_bootstrap_samples (int): The number of bootstrap samples to generate.
        confidence_level (float): The desired confidence level (between 0 and 1).

    Returns:
        tuple: Lower and upper bounds of the confidence interval.

    """
    # Convert the data to a NumPy array
    data = np.array(data)

    # Create an array to store the bootstrap sample statistics
    bootstrap_samples = np.zeros(num_bootstrap_samples)

    # Perform bootstrapping
    for i in range(num_bootstrap_samples):
        # Generate a bootstrap sample by randomly sampling with replacement from the original data
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)

        # Calculate the statistic of interest on the bootstrap sample
        bootstrap_statistic = np.mean(bootstrap_sample)

        # Store the bootstrap statistic
        bootstrap_samples[i] = bootstrap_statistic

    # Calculate the lower and upper percentiles of the bootstrap samples
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile

    lower_bound = np.percentile(bootstrap_samples, lower_percentile * 100)
    upper_bound = np.percentile(bootstrap_samples, upper_percentile * 100)
    print(f"Confidence interval: [{lower_bound:.2f}, {upper_bound:.2f}]")
    return lower_bound, upper_bound

# Print the confidence interval
def violinplot(data: pd.DataFrame, category_column: str, numeric_column: str, xlabel: str = 'Category',
                     ylabel: str = "Numeric values", title: str = "Title" ,figsize: Tuple[float, float] = (8,8)) -> None:
        """
    Creates a violin plot for the given dataframe using the specified category column and numeric column.

    Args:
        data (pd.DataFrame): The input dataframe to plot.
        category_column (str): The name of the column to use as the categorical variable.
        numeric_column (str): The name of the column to use as the numeric variable.
        xlabel (str, optional): The label for the x-axis. Defaults to 'Category'.
        ylabel (str, optional): The label for the y-axis. Defaults to 'Numeric values'.
        title (str, optional): The title for the plot. Defaults to 'Title'.
        figsize (Tuple[float, float], optional): The size of the figure.


    Returns:
        Violin plot
    """
        fig = plt.figure(figsize = figsize)
        ax = fig.gca()
        sns.set_style('whitegrid')
        # cmap naudojama spalvu palete
        sns.violinplot(x = category_column, y = numeric_column, data = data, ax = ax)


        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
    
  
    

def kdePlot(datacolumn: pd.Series, xlabel: str = 'Size', ylabel: str = 'Density', 
                    title: str = "Kernel density plot", figsize: Tuple[int, int] = (10,10) ) -> None:
        """
    Creates a kernel density plot for the given pandas series.

    Args:
        datacolumn (pd.Series): The input data to plot.
        xlabel (str, optional): The label for the x-axis. Defaults to 'Size'.
        ylabel (str, optional): The label for the y-axis. Defaults to 'Density'.
        title (str, optional): The title for the plot. Defaults to 'Kernel density plot'.
        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (10,10).

    Returns:
        Kernel density plot
    """
        fig = plt.figure(figsize = figsize)
        ax = fig.gca()
        sns.set_style('whitegrid')
        # vizualizuoja, variklio dydi pagal kuro tipa
        sns.kdeplot(datacolumn,ax=ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)   
    
    
    
@staticmethod
def OneD_Bar_Sublots (counts_list: List[pd.Series], subtitle: str = "Bar chart subplots", 
                            xlabel: str = 'Category', ylabel: str = 'Values',  figsize = (10,10)) -> None:
        """
    Creates multiple bar chart subplots for the given list of pandas series.

    Args:
        counts_list (List[pd.Series]): The list of data to plot.
        subtitle (str, optional): The subtitle for the plot. Defaults to 'Bar chart subplots'.
        xlabel (str, optional): The label for the x-axis. Defaults to 'Category'.
        ylabel (str, optional): The label for the y-axis. Defaults to 'Values'.

    Returns:
        One dimension barchart subplots
    """
        fig, axs = plt.subplots(1, len(counts_list), figsize=figsize)
        for idx, i in enumerate(counts_list):
            axs[idx].bar(i.index, i.values)
            axs[idx].set_title(input(f'Set title for {i.name} subplot (they are in same order as in your list):  '))
           

            
        for ax in axs.flat:
            for p in ax.patches:
                ax.text(
                p.get_x() + p.get_width() / 2,
                p.get_height(),
                p.get_height(),
                ha="center",
                va="bottom",
                
        )

        fig.suptitle(subtitle)
        plt.xlabel(xlabel )
        plt.ylabel(ylabel)

        # Show the plot
        plt.show()

def histogram(dataframe_column: pd.Series, title: str = "Title", xlabel: str = "Sizes",
              ylabel: str = "Amount", figsize: tuple = (10, 10), rotation: int = 0) -> None:
    """
    This function creates a histogram plot of a given pandas Series.

    Args:
        dataframe_column (pd.Series): A pandas Series object to be plotted.
        title (str, optional): The title of the histogram. Defaults to "Title".
        xlabel (str, optional): The label of the x-axis. Defaults to "Sizes".
        ylabel (str, optional): The label of the y-axis. Defaults to "Amount".
        figsize (tuple, optional): The size of the figure. Defaults to (10, 10).
        rotation (int, optional): The rotation of the x-tick labels. Defaults to 0.

    Returns:
         None
    """
    values = dataframe_column.values
    fig = plt.figure(figsize=figsize)
    plt.rotation = rotation
    ax = fig.gca()
    dataframe_column.plot.hist(ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
def oneD_piechart_subplots(data_list: list[pd.Series], subtitle: str = "Pie chart subplots", figsize: tuple[int, int] = (15, 10)) -> None:
        """
        Plot one-dimensional pie chart subplots.

        Parameters:
        data_list (list[pd.Series]): A list of Pandas Series objects containing the data to plot.
        subtitle (str): The title of the plot. Default is 'Pie chart subplots'.
        figsize (tuple[int, int]): The size of the figure. Default is (15, 10).

        Returns:
        None

        Raises:
        ValueError: If `data_list` is empty.
        """
        fig, axs = plt.subplots(1, len(data_list), figsize=figsize)
        for idx, i in enumerate(data_list):
            axs[idx].pie(
                i.values.astype(float),
                labels=i.index,
                autopct="%1.1f%%",
        )   
            axs[idx].set_title(
                input(
                f"Set title for {i.name} subplot (they are in same order as in your list):  "
            )
        )
        # Plot a pie chart on each of the subplots

        # Add a title to the figure
        fig.suptitle(subtitle)

        # Show the plot
        plt.show()

def calculate_mean_last_five_results_for_all_matches(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the mean of the last five results for both home and away teams for all matches.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing match data.

    Returns:
        pd.DataFrame: The original DataFrame with additional columns for mean results.
    """
    # Create new columns to store the mean results for home and away teams
    dataframe["mean_result_home"] = np.nan
    dataframe["mean_result_away"] = np.nan

    for index, row in dataframe.iterrows():
        home_team_api_id = row["home_team_api_id"]
        away_team_api_id = row["away_team_api_id"]

        # Calculate the mean result for the home team
        mean_result_home, _ = calculate_mean_last_five_results(dataframe, home_team_api_id, "result_home", index)

        # Calculate the mean result for the away team
        mean_result_away, _ = calculate_mean_last_five_results(dataframe, away_team_api_id, "result_away", index)

        # Update the DataFrame with the mean results
        dataframe.at[index, "mean_result_home"] = mean_result_home
        dataframe.at[index, "mean_result_away"] = mean_result_away

    return dataframe

def calculate_mean_last_five_results(dataframe: pd.DataFrame, team_api_id: int, result_column: str, current_match_index: int) -> Tuple[float, list]:
    """
    Calculate the mean of the last five results for a specific team.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing match data.
        team_api_id (int): The API ID of the team for which to calculate the mean.
        result_column (str): The name of the column containing match results.
        current_match_index (int): The index of the current match being processed.

    Returns:
        Tuple[float, list]: A tuple containing the mean result (float) and a list of meanings for the last five results.
    """
    # Filter the DataFrame for matches involving the specified team
    team_matches = dataframe[(dataframe["home_team_api_id"] == team_api_id) | (dataframe["away_team_api_id"] == team_api_id)]
    
    # Exclude the current match from the calculations
    team_matches = team_matches[team_matches.index != current_match_index]
    
    # Sort the matches by index (assuming the DataFrame is sorted by date)
    team_matches = team_matches.sort_index(ascending=False)
    
    # Extract the last five results for the team (or all available if fewer than five)
    last_five_results = team_matches[result_column].head(5).values
    
    # Map the result codes to their corresponding meanings (1: Loss, 2: Draw, 3: Win)
    result_meanings = {1: "Loss", 2: "Draw", 3: "Win"}
    last_five_results_meaning = [result_meanings[result_code] for result_code in last_five_results]
    
    # Calculate the mean of the results (1: Loss, 2: Draw, 3: Win)
    mean_result = np.mean(last_five_results)
    
    return mean_result, last_five_results_meaning

def calculate_mean_last_five_home_results(dataframe: pd.DataFrame, team_api_id: int, result_column: str, current_match_index: int) -> Tuple[float, list]:
    """
    Calculate the mean of the last five home results for a specific team.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing match data.
        team_api_id (int): The API ID of the team for which to calculate the mean.
        result_column (str): The name of the column containing match results.
        current_match_index (int): The index of the current match being processed.

    Returns:
        Tuple[float, list]: A tuple containing the mean result (float) and a list of meanings for the last five home results.
    """
    # Filter the DataFrame for matches where the team is the home team
    home_matches = dataframe[dataframe["home_team_api_id"] == team_api_id]
    
    # Exclude the current match from the calculations
    home_matches = home_matches[home_matches.index != current_match_index]
    
    # Sort the matches by index (assuming the DataFrame is sorted by date)
    home_matches = home_matches.sort_index(ascending=False)
    
    # Extract the last five home results for the team (or all available if fewer than five)
    last_five_home_results = home_matches[result_column].head(5).values
    
    # Map the result codes to their corresponding meanings (1: Loss, 2: Draw, 3: Win)
    result_meanings = {1: "Loss", 2: "Draw", 3: "Win"}
    last_five_home_results_meaning = [result_meanings[result_code] for result_code in last_five_home_results]
    
    # Calculate the mean of the home results (1: Loss, 2: Draw, 3: Win)
    mean_home_result = np.mean(last_five_home_results)
    
    return mean_home_result, last_five_home_results_meaning

def calculate_mean_last_five_away_results(dataframe: pd.DataFrame, team_api_id: int, result_column: str, current_match_index: int) -> Tuple[float, list]:
    """
    Calculate the mean of the last five away results for a specific team.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing match data.
        team_api_id (int): The API ID of the team for which to calculate the mean.
        result_column (str): The name of the column containing match results.
        current_match_index (int): The index of the current match being processed.

    Returns:
        Tuple[float, list]: A tuple containing the mean result (float) and a list of meanings for the last five away results.
    """
    # Filter the DataFrame for matches where the team is the away team
    away_matches = dataframe[dataframe["away_team_api_id"] == team_api_id]
    
    # Exclude the current match from the calculations
    away_matches = away_matches[away_matches.index != current_match_index]
    
    # Sort the matches by index (assuming the DataFrame is sorted by date)
    away_matches = away_matches.sort_index(ascending=False)
    
    # Extract the last five away results for the team (or all available if fewer than five)
    last_five_away_results = away_matches[result_column].head(5).values
    
    # Map the result codes to their corresponding meanings (1: Loss, 2: Draw, 3: Win)
    result_meanings = {1: "Loss", 2: "Draw", 3: "Win"}
    last_five_away_results_meaning = [result_meanings[result_code] for result_code in last_five_away_results]
    
    # Calculate the mean of the away results (1: Loss, 2: Draw, 3: Win)
    mean_away_result = np.mean(last_five_away_results)
    
    return mean_away_result, last_five_away_results_meaning

def getTeamResult(row):
    if row["winning_team"] == "1":
        home_team_result = "Win"
        away_team_result = "Loss"
    elif row["winning_team"] == "3":
        home_team_result = "Loss"
        away_team_result = "Win"
    else:
        home_team_result = "Draw"
        away_team_result = "Draw"

    return [home_team_result, away_team_result]
    
def calculate_rolling_means(df, feature, window_size=5):
    """
    Calculate rolling means for a specified feature for both home and away games.

    Args:
        df (pd.DataFrame): The DataFrame containing match data.
        feature (str): The name of the feature for which to calculate rolling means.
        window_size (int): The size of the rolling window (default is 5).

    Returns:
        pd.DataFrame: The DataFrame with additional columns for rolling means.
    """
    # Calculate the rolling mean for home games
    df[feature + "_home_game"] = df.groupby("home_team_api_id")[feature].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )

    # Calculate the rolling mean for away games
    df[feature + "_away_game"] = df.groupby("away_team_api_id")[feature].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )

    # Calculate the rolling mean difference between home and away teams
    df[feature + "_RMean_Diff"] = (
        df[feature + "_home_game"] - df[feature + "_away_game"]
    )

    return df

def delete_player_columns(df, start_player=1, end_player=11):
    for i in range(start_player, end_player + 1):
        columns_to_delete = [col for col in df.columns if col.endswith(f"player_{i}")]
        df.drop(columns=columns_to_delete, inplace=True)
    return df


def calculate_team_mean_cat_data(df, team, feature_name, start_player=1, end_player=11):
    # Generate the list of player columns based on the specified range
    player_columns = [
        f"{feature_name}_{team}_player_{i}" for i in range(start_player, end_player + 1)
    ]

    # Convert the player columns to numeric (replace non-numeric values with NaN)
    df[player_columns] = df[player_columns].apply(pd.to_numeric, errors="coerce")

    # Calculate the mean for the specified player columns
    df[f"{team}_team_mean_{feature_name}"] = df[player_columns].mean(axis=1)

    # Drop the player columns that were used to calculate the mean
    df.drop(columns=player_columns, inplace=True)

    # Convert the "defensive_work_rate" columns to numeric values based on player number
    encoding_map = {"low": 1, "medium": 2, "high": 3}
    for i in range(start_player, end_player + 1):
        defensive_work_rate_column = f"{feature_name}_{team}_player_{i}"
        # Check if the column exists before replacing values
        if defensive_work_rate_column in df.columns:
            df[defensive_work_rate_column] = df[defensive_work_rate_column].replace(
                encoding_map
            )

    return df
def calculate_team_mean(df, team, feature_name, start_player=1, end_player=11):
    player_columns = [f"{team}_player_{i}" for i in range(start_player, end_player + 1)]
    player_feature_columns = [
        f"{feature_name}_{team}_player_{i}" for i in range(start_player, end_player + 1)
    ]

    # Check if all player feature columns exist
    if all(col in df.columns for col in player_feature_columns):
        df[f"{team}_team_mean_{feature_name}"] = df[player_feature_columns].mean(axis=1)
        df.drop(columns=player_feature_columns, inplace=True)
    else:
        print("Player feature columns do not exist in the DataFrame.")

    return df


def pie_count_subplot_single(data_column, title):
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot pie chart
    data_column.value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
    ax[0].set_title(title)
    ax[0].set_ylabel('')
    
    # Plot countplot
    sns.countplot(x=data_column.name, data=data_column.to_frame(), ax=ax[1])
    ax[1].set_title(title)
    
    plt.show()

def bar_hued_barchart(data: pd.DataFrame, column: str, hue_column: str, title1: str = 'title1', title2: str = 'title2', title: str = 'title', xlabel1: str = 'xlabel1', xlabel2: str = 'xlabel2'):
    """
    Plot a bar chart and a hued bar chart for the specified columns with customizable titles and x-axis labels.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column to be plotted on the x-axis.
    hue_column (str): The column to be used for hue in the countplot.
    title1 (str): The title for the first subplot (bar chart). Default is 'title1'.
    title2 (str): The title for the second subplot (countplot). Default is 'title2'.
    title (str): The general title for the plots. Default is 'title'.
    xlabel1 (str): The x-axis label for the first subplot. Default is 'xlabel1'.
    xlabel2 (str): The x-axis label for the second subplot. Default is 'xlabel2'.
    """
    f, ax = plt.subplots(1, 2, figsize=(18, 8))

    # Plot bar chart
    data[[column, hue_column]].groupby([column]).mean().plot.bar(ax=ax[0])
    ax[0].set_title(title1)
    ax[0].set_xlabel(xlabel1)

    # Plot countplot
    sns.countplot(x=column, hue=hue_column, data=data, ax=ax[1])
    ax[1].set_title(title2)
    ax[1].set_xlabel(xlabel2)

    plt.suptitle(title)
    plt.show()

def plot_count_and_hue(data: pd.DataFrame, x_column: str, hue_column: str):
    """
    Plot two countplots: one without hue and one with hue.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    x_column (str): The categorical variable to be plotted on the x-axis.
    hue_column (str): The categorical variable to be used for hue.
    """
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot countplot without hue
    sns.countplot(x=x_column, data=data, ax=axes[0])
    axes[0].set_title(f'Countplot of {x_column}')

    # Plot countplot with hue
    sns.countplot(x=x_column, hue=hue_column, data=data, ax=axes[1])
    axes[1].set_title(f'Countplot of {x_column} with Hue {hue_column}')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

def plot_proportion_stacked_bar(data: pd.DataFrame, x_column: str, hue_column: str,title:str):
    """
    Plot a stacked bar chart to visualize the proportion of hued counts.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    x_column (str): The categorical variable to be plotted on the x-axis.
    hue_column (str): The categorical variable to be used for hue.
    """
    # Create a contingency table
    contingency_table = pd.crosstab(data[x_column], data[hue_column], normalize='index')

    ax = contingency_table.plot(kind='bar', stacked=True, rot=0)
    ax.legend(title=hue_column, bbox_to_anchor=(1, 1.02), loc='upper left')

    # add annotations if desired
    for c in ax.containers:
    
    # set the bar label
        ax.bar_label(c, label_type='center')
    ax.set_title(title)    

def plot_bar_and_stacked_bar(data: pd.DataFrame, x_column: str, hue_column: str, title: str=None, subplot_title: str = None, main_title: str = None,figsize:tuple = (12,8)):
    """
    Plot a bar chart of data[x_column] followed by a stacked bar chart to visualize the proportion of hued counts.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    x_column (str): The categorical variable to be plotted on the x-axis.
    hue_column (str): The categorical variable to be used for hue.
    title (str): The title for the plot.
    subplot_title (str, optional): The title for the subplot.
    main_title (str, optional): The main title for the plot.

    Returns:
    tuple[str, str, str]: The titles of the bar chart, stacked bar chart, and subplot.
    """
    # Check if there are multiple categories in x_column
    if len(data[x_column].unique()) > 1:
        # Create a figure with subplots for both charts
        fig, axes = plt.subplots(1, 2, figsize = figsize)
        
        # Create a bar chart for data[x_column]
        data[x_column].value_counts().plot(kind='bar', ax=axes[0], rot=0)
        axes[0].set_title(f'Bar Chart of {x_column}')
        
        # Create a contingency table for the stacked bar chart
        contingency_table = pd.crosstab(data[x_column], data[hue_column], normalize='index')
        
        ax = contingency_table.plot(kind='bar', stacked=True, rot=0, ax=axes[1])
        ax.legend(title='Stacked Barchart of ' + hue_column, bbox_to_anchor=(1, 1.02), loc='upper left')
    else:
        # Only one category in x_column, so create only the stacked bar chart
        plt.figure(figsize=figsize)
        contingency_table = pd.crosstab(data[x_column], data[hue_column], normalize='index')
        ax = contingency_table.plot(kind='bar', stacked=True, rot=0)
        ax.legend(title='Stacked Barchart of ' + hue_column, bbox_to_anchor=(1, 1.02), loc='upper left')
        # add annotations if desired
    for c in ax.containers:
    
    # set the bar label
        ax.bar_label(c, label_type='center')

    # Set the title of the plot
    if main_title is not None:
        plt.suptitle(main_title, fontsize=16)

    # Adjust layout for better spacing between subplots
    plt.tight_layout()
    plt.title(' Stacked barchart in comparison with'  + hue_column  )
    plt.show()

def lasso_classifier(X_train,y_train,X_test,y_test,X):
    
    lasso_classifier = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)
    lasso_classifier.fit(X_train, y_train)


    y_pred = lasso_classifier.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


    lasso_coefficients = lasso_classifier.coef_[0]


    lasso_abs_coefficients = np.abs(lasso_coefficients)

    top_20_lasso_indices = np.argsort(lasso_abs_coefficients)[-20:]


    top_20_lasso_feature_names = X.columns[top_20_lasso_indices]


    top_20_lasso_coefficients = lasso_coefficients[top_20_lasso_indices]

    # Create a bar plot to visualize the top 20 most important features for Lasso
    plt.figure(figsize=(12, 8))
    plt.barh(top_20_lasso_feature_names, top_20_lasso_coefficients)
    plt.xlabel("Coefficient Value (Lasso)")
    plt.title("Most Important Features - Lasso")
    plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
    plt.show()    

def ridge_classifier(X_train,y_train,X_test,y_test,X):
    ridge_classifier = LogisticRegression(penalty="l2", solver="liblinear", random_state=42)
    ridge_classifier.fit(X_train, y_train)

   
    y_pred = ridge_classifier.predict(X_test)

   
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

   
    ridge_coefficients = ridge_classifier.coef_[0]

    
    ridge_abs_coefficients = np.abs(ridge_coefficients)

    
    top_20_ridge_indices = np.argsort(ridge_abs_coefficients)[-20:]

   
    top_20_ridge_feature_names = X.columns[top_20_ridge_indices]

    
    top_20_ridge_coefficients = ridge_coefficients[top_20_ridge_indices]

    
    plt.figure(figsize=(12, 8))
    plt.barh(top_20_ridge_feature_names, top_20_ridge_coefficients)
    plt.xlabel("Coefficient Value (Ridge)")
    plt.title("Top 20 Most Important Features - Ridge")
    plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
    plt.show()  

def is_binary(series):
    unique_values = series.unique()
    return len(unique_values) == 2 and set(unique_values) == {0, 1}
