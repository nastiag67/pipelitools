import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

from pipelitools import utils as u


def test_outliers():
    print('test_outliers: ok')


class Outliers:
    """Shows outliers calculated by one of the available functions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe (icluding features and responses).

    """

    def __init__(self, df):
        self.df = df

    def _z_score(self, columns, threshold=3):
        """Detects outliers based on z-score.

        Parameters:
        ----------
        columns : str
            A string of columns which will be analysed together using z-score.

        threshold : int, default=3
            Threshold against which the outliers are detected.

        Returns
        ----------
        df_outliers_clean : dataframe
            Dataframe without outliers.

        df_outliers : dataframe
            Dataframe of outliers.

        """
        # remove outliers based on chosen columns
        df_selected = self.df[columns].copy()

        # remove outliers
        z = np.abs(stats.zscore(df_selected))

        df_clean = self.df[(z < threshold).all(axis=1)]

        # get outliers df
        df_outliers = self.df[~self.df.index.isin(df_clean.index)]

        return df_clean, df_outliers

    def _IQR(self, columns, q1=0.25):
        """Detects outliers based on interquartile range (IQR).

        Parameters:
        ----------
        columns : str
            A string of columns which will be analysed together using IQR.

        q1 : float, default=0.25
            Threshold against which the outliers are detected.

        Returns
        ----------
        df_outliers_clean : dataframe
            Dataframe without outliers.

        df_outliers : dataframe
            Dataframe of outliers.

        """
        # remove outliers based on chosen columns
        # print(columns)
        df_selected = self.df[columns]

        # remove outliers
        Q1 = df_selected.quantile(q1)
        Q3 = df_selected.quantile(1 - q1)
        IQR = Q3 - Q1

        df_clean = self.df[~((df_selected < (Q1 - 1.5 * IQR)) | (df_selected > (Q3 + 1.5 * IQR))).any(axis=1)]

        # get outliers df
        df_outliers = self.df[~self.df.index.isin(df_clean.index)]

        return df_clean, df_outliers

    def _plot(self, columns, df_clean, df_outliers, plot_cols=4):
        """Plots the dataframe and marks the outliers by a red cross.

        Parameters:
        ----------
        columns : str
            A string of columns which will be plotted.

        df_clean : dataframe
            Dataframe without outliers.

        df_outliers : dataframe
            Dataframe of outliers.

        plot_cols : int, default=6
            Determines how many columns the plots will form.

        """
        plt.style.use('seaborn-white')

        if plot_cols > len(columns) - 2:
            u.log(u.yellow('ERROR: '), f"Can't use more than {len(columns) - 2} columns in one row.")
            plot_cols = len(columns) - 2

        # figure size = (width,height)
        f1 = plt.figure(figsize=(30, len(columns) * 3))

        total_plots = len(columns)
        rows = total_plots - plot_cols

        for idx, y in enumerate(columns):
            idx += 1
            ax1 = f1.add_subplot(rows, plot_cols, idx)
            sns.regplot(x=df_clean.index,
                        y=y,
                        data=df_clean,
                        scatter=True,
                        fit_reg=False,
                        color='lightblue',
                        )
            sns.regplot(x=df_outliers.index,
                        y=y,
                        data=df_outliers,
                        scatter=True,
                        fit_reg=False,
                        marker='x',
                        color='red',
                        )

    def show_outliers(self, columns, how='z_score', show_plot=False, **kwargs):
        """Detects outliers using one of the available methods.

        Parameters:
        ----------
        df : dataframe
            Feature dataframe.

        columns : list
            A list of columns which will be analysed together.

        how : str, default=z_score
            Method using which the outliers are detected.

        show_plot : bool, default=False
            True if need to see the plot of the data with the marked outliers.

        **kwargs
            Specifies extra arguments which may be necessary for one of the methods of finding outliers:

            threshold : int, default=3
                True if need to return all the formats of the columns.

            q1 : float, default=0.25
                True if need to return all the formats of the columns.

        Returns
        ----------
        df_clean : dataframe
            Dataframe without outliers.

        df_outliers : dataframe
            Dataframe of outliers.

        df : dataframe
            Original dataframe with outliers.
            Contains a new column called 'outliers' (bool) where the outliers are flagged (True if outlier).

        """
        if how == 'z_score':
            assert 'threshold' in kwargs, 'To use z-score method, threshold must be specified (default = 3)'
            df_clean, df_outliers = self._z_score(columns, kwargs['threshold'])
        elif how == 'IQR':
            assert 'q1' in kwargs, 'To use z-score method, q1 must be specified (default = 0.25)'
            df_clean, df_outliers = self._IQR(columns, kwargs['q1'])
        else:
            raise AttributeError('Unknown outlier detection method. Existing methods: z_score, IQR')
        df = self.df.copy()
        df['outliers'] = df.index.isin(df_outliers.index).copy()

        # print('-'*100)
        if show_plot:
            self._plot(columns, df_clean, df_outliers)

        return df_clean, df_outliers, df


if __name__ == '__main__':
    test_outliers()