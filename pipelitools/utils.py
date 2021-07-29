import matplotlib.pyplot as plt

# import xlwings as xw
from functools import wraps
import datetime
import pandas as pd
from calendar import monthrange
import os
import errno


def to_dates(df, cols):
    """ Changes column format to datetime.

    Parameters:
    ----------
    df : dataframe
        Dataframe with columns which are falsely not recognised as datetime.

    cols : list
        list of columns, formats of which need to be corrected.

    Returns
    ----------
    df : dataframe with corrected column formats

    """
    for col in cols:
        df[col] = pd.to_datetime(df[col])
    return df


def get_new_date(date, delta_days):
    '''
    INPUT:
        date (str) - end date in format 'DD.MM.YYYY';
        days_back (int) - how may days to substract from the date.
    RETURNS:
        beginning of the period = date - days_back
    '''
    if len(date) != 10:
        raise ValueError("date format is wrong, expected format: 'DD.MM.YYYY'")

    d = int(date[0:2])
    m = int(date[3:5])
    y = int(date[6:10])

    dt = datetime.datetime(y, m, d)

    dt_end = dt.date() + datetime.timedelta(days=delta_days)

    if len(str(dt_end.month)) == 2:
        if len(str(dt_end.day)) == 2:
            b = str(dt_end.day) + '.' + str(dt_end.month) + '.' + str(dt_end.year)
        else:
            b = '0' + str(dt_end.day) + '.' + str(dt_end.month) + '.' + str(dt_end.year)
    else:
        if len(str(dt_end.day)) == 2:
            b = str(dt_end.day) + '.0' + str(dt_end.month) + '.' + str(dt_end.year)
        else:
            b = '0' + str(dt_end.day) + '.0' + str(dt_end.month) + '.' + str(dt_end.year)

    return b


def get_first_day(date):
    '''
    INPUT:
        date (str) - any date in format 'DD.MM.YYYY';
    RETURNS:
        first day of the month
    '''
    if len(date) != 10:
        raise ValueError("date format is wrong, expected format: 'DD.MM.YYYY'")

    d = int(date[0:2])
    m = int(date[3:5])
    y = int(date[6:10])

    dt = datetime.datetime(y, m, d)
    dt_new = dt.replace(day=1)

    if len(str(dt_new.month)) == 2:
        result = '0' + str(dt_new.day) + '.' + str(dt_new.month) + '.' + str(dt_new.year)
    else:
        result = '0' + str(dt_new.day) + '.0' + str(dt_new.month) + '.' + str(dt_new.year)

    return result


def offset_end_date(date, mths_offset):
    '''
    INPUT:
        date (str) - end date in format 'DD.MM.YYYY';
        mths_offset (int) - how may months to offset from the date.
    RETURNS:
        end of the month which is offset by mths_offset from initial date
    '''
    if len(date) != 10:
        raise ValueError("date format is wrong, expected format: 'DD.MM.YYYY'")

    if mths_offset == 0:
        return date

    d = int(date[0:2])
    m = int(date[3:5])
    y = int(date[6:10])

    dt = datetime.datetime(y, m, d)

    dt_end = dt + pd.DateOffset(months=mths_offset)

    dt_end_day = monthrange(dt_end.year, dt_end.month)[1]

    if len(str(dt_end.month)) == 2:
        result = str(dt_end_day) + '.' + str(dt_end.month) + '.' + str(dt_end.year)
    else:
        result = str(dt_end_day) + '.0' + str(dt_end.month) + '.' + str(dt_end.year)

    return result


def offset_first_date(date, mths_offset):
    '''
    INPUT:
        date (str) - end date in format 'DD.MM.YYYY';
        mths_offset (int) - how may months to offset from the date.
    RETURNS:
        end of the month which is offset by mths_offset from initial date
    '''
    if len(date) != 10:
        raise ValueError("date format is wrong, expected format: 'DD.MM.YYYY'")

    d = int(date[0:2])
    m = int(date[3:5])
    y = int(date[6:10])

    dt = datetime.datetime(y, m, d)

    dt_first = dt + pd.DateOffset(months=mths_offset)

    if len(str(dt_first.month)) == 2:
        result = '0' + str(dt_first.day) + '.' + str(dt_first.month) + '.' + str(dt_first.year)
    else:
        result = '0' + str(dt_first.day) + '.0' + str(dt_first.month) + '.' + str(dt_first.year)

    return result


def columns_list(df, dictionary=False):
    '''
    INPUT:
        df - dataframe;
        dictionary - if the output should be in form of dicitonary
    RETURNS:
        list of column names
    '''
    if dictionary == False:
        for i in df.columns:
            print("'" + i + "'" + ",")
    else:
        for i in df.columns:
            print("'" + i + "'" + ":" + "'" + "'" + ",")


def text_format(bold=False, underline=False, red=False, green=False, yellow=False):
    
    tags = []
    
    if bold:
        tag='\033[1m'
        tags.append(tag)
        
    if underline:
        tag='\033[4m'
        tags.append(tag)

    if red:
        tag='\033[91m'
        tags.append(tag)

    if green:
        tag='\033[92m'
        tags.append(tag)

    if yellow:
        tag = '\033[93m'
        tags.append(tag)
        
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            msg = func(*args, **kwargs)
            return ('{}'*len(tags)).format(*tags) + msg
        # Return the decorated function
        return wrapper
    # Return the decorator
    return decorator


def export_str(output, filename):
    """
    To export & create folder in current directory, do filename="./classification_report/name.txt"
    """
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(filename, "w") as f:
        f.write(output)


# def export_xls(output, filename, sheetname, after=None):
#     try:
#         # UPDATE EXCEL
#         wb = xw.Book(filename+'.xlsx')
#         try:
#             sht1 = wb.sheets[sheetname]
#         except:
#             if after!=None:
#                 wb.sheets.add(name=sheetname)
#             else:
#                 wb.sheets.add(name=sheetname, after = after)
#             sht1 = wb.sheets[sheetname]
#         sht1.range('A1').value = output
#     except FileNotFoundError:
#         # EXPORT
#         writer = pd.ExcelWriter(filename + '.xlsx')
#         output.to_excel(writer, sheet_name=sheetname)
#         writer.save()


def export_pic(fig_name, folder=None):
    if folder:
        if os.path.exists(folder) is False:
            os.mkdir(folder)
        plt.savefig(f"./{folder}/{fig_name}.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"{fig_name}.png", dpi=300, bbox_inches='tight')


# LOGS
def log(*m):
    print(" ".join(map(str, m)))


def black(s):
    return '\033[1;30m%s\033[m' % s


def green(s):
    return '\033[1;32m%s\033[m' % s


def red(s):
    return '\033[1;31m%s\033[m' % s


def yellow(s):
    return '\033[1;33m%s\033[m' % s


def check(expression, text=None):
    if expression:
        return log(green('PASSED.'), text)
    else:
        return log(red('ERROR.'), text)
