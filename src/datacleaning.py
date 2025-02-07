import pandas as pd

class DataCleaner:
    """
    A utility class for cleaning pandas DataFrames.

    Methods:
        drop_column: Drops a specified column.
        drop_duplicates: Removes duplicate rows.
        drop_na: Removes rows with null values.
    """    
    def col_rename(self, df):
        """
        Renames the a columns inside the Twee Df.
        
        Parameters:
            an input pandas Dataframe
        
        Returns:
            pd.DataFrame: the dataframe with specified column removed
        """
        df.rename(columns={list(df)[0]:'TweetID',
                       list(df)[1]:'Entity',
                      list(df)[2]:'Sentiment',
                      list(df)[3]:'Tweet'},inplace=True)
        print('Columns successfully renamed')
        return df
    
    def drop_column(self, df, col):
        """
        Drops a column previded. Suggesion is to drop the Tweet ID col
        
        Parameters:
            an input pandas Dataframe
        
        Returns:
            pd.DataFrame: the dataframe with specified column removed
        """
        df.drop(columns = col, inplace=True)
        print('Unwanted_columns successfully removed')
        return df
        
    def drop_duplicate(self, df):
        """
        Purpose:
            Drops duplicate rows
        Returns:
            Cleaned dataframe
        """
        df = df.drop_duplicates(inplace=True)
        print('Duplicate columns successfully removed')
        return df

    def drop_na(self, df):
        """
        Purpose:
            Drops null values in the dataframe
        Retruns:
            Cleaned dataframe
        """
        df = df.dropna()
        print('Null values successfully removed')
        return df

