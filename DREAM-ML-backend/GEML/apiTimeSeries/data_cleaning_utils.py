# Copyright (C) 2025 Leonardo Espinoza Ortiz <leonardo.espinoza.o@usach.cl>
#
# This file is part of DREAM ML.
#
# DREAM ML is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DREAM ML is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DREAM ML. If not, see <https://www.gnu.org/licenses/>.
import pandas as pd
import numpy as np
import shutil
from datetime import datetime, timedelta
from dateutil import parser
import warnings
import json


###################################################################################
############################### AUXILIARY FUNCTIONS ###############################
###################################################################################

# MANDATORY CLEANING METHODS #

def clean_column_names(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, dict]:
    df.columns = df.columns.str.strip()
    return df, report

def remove_whitespace_from_df_data(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, dict]:
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df, report

def replace_empty_values_with_nan(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, dict]:
    df = df.replace(r'^\s*$', np.nan, regex=True)
    return df, report

def convert_to_numeric_columns(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, dict]:
    # 3. Convertir columnas numéricas mal interpretadas
    converted_cols = []
    for col in df.columns:
        if df[col].dropna().apply(lambda x: str(x).replace('.', '', 1).isdigit()).all():
            df[col] = pd.to_numeric(df[col], errors='coerce')
            converted_cols.append(col)
    report["converted_to_numeric"] = converted_cols
    return df, report

def fill_categorical_missing(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, dict]:
   # 4. Rellenar valores faltantes en columnas categóricas con 'vacio'
   categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
   cat_missing = {}
   for col in categorical_cols:
    missing_count = int(df[col].isna().sum())
    if missing_count > 0:
        df[col] = df[col].fillna('vacio')
        cat_missing[col] = missing_count
   report["categorical_missing_filled"] = cat_missing
   return df, report

def remove_empty_columns(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, dict]:
    # 5. Eliminar columnas con solo valores nulos
    cols_before = list(df.columns)
    df = df.dropna(axis=1, how='all')
    cols_after = list(df.columns)
    removed_columns = list(set(cols_before) - set(cols_after))
    report["columns_removed_all_na"] = removed_columns
    return df, report

# OPTIONAL METHODS #

def drop_duplicates(df: pd.DataFrame, report: dict,
                    include: bool) -> tuple[pd.DataFrame, dict]:
    if include:
        before_dup = df.shape[0]
        # Fix: Sort before drop_duplicates for deterministic duplicate removal
        # Without sorting, which duplicate is kept varies by DataFrame memory layout
        # Sorting ensures consistent behavior across runs
        df = df.sort_values(by=df.columns.tolist()).reset_index(drop=True)
        df = df.drop_duplicates(keep='first')
        after_dup = df.shape[0]
        report["duplicates_removed"] = int(before_dup - after_dup)
    else:
        report["duplicates_removed"] = 0
    return df, report

def fill_missing_numeric_values(df:pd.DataFrame, report:dict, 
                                method: str, 
                                value: float) -> tuple[pd.DataFrame, dict]:
    # Rellenar valores faltantes en columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report["numeric_missing_before"] = df[numeric_cols].isna().sum().to_dict()
    
    if method == "eliminar":
        before_drop = df.shape[0]
        df = df.dropna(subset=numeric_cols)
        after_drop = df.shape[0]
        report["numeric_rows_dropped_due_to_na"] = int(before_drop - after_drop)
    elif method == "dejar":
        # No se realiza cambio en los NaN
        report["numeric_missing_after"] = df[numeric_cols].isna().sum().to_dict()
    else:
        imputations = {}
        for col in numeric_cols:
            missing_count = int(df[col].isna().sum())
            if missing_count > 0:
                if method == "media":
                    fill_value = df[col].mean()
                    df[col] = df[col].fillna(fill_value)
                    imputations[col] = {"filled_with": "mean", "missing_count": missing_count, "fill_value": fill_value}
                elif method == "valor" and value is not None:
                    df[col] = df[col].fillna(value)
                    imputations[col] = {"filled_with": value, "missing_count": missing_count}
        report["numeric_imputations"] = imputations
    return df, report

def filter_outliers(df:pd.DataFrame, report:dict) -> tuple[pd.DataFrame, dict]:
    # 7. Filtrar valores atípicos (opcional)
    outliers_removed = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].notna().sum() > 1:
            before_filter = df.shape[0]
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2.5 * IQR
            upper_bound = Q3 + 2.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            after_filter = df.shape[0]
            outliers_removed[col] = int(before_filter - after_filter)
    report["outliers_removed"] = outliers_removed
    return df, report
def standardize_date_to_utc(df: pd.DataFrame, report: dict, date_column: str, imputation_strategy: str = "mean_timedelta") -> tuple[pd.DataFrame, dict]:
      """
      Enhanced UTC standardization with invalid date handling.
      """
      function_name = "standardize_date_to_utc"

      try:
          if date_column not in df.columns:
              report[function_name] = False
              report[f"{function_name}_error"] = f"Column '{date_column}' not found"
              return df, report

          original_null_count = df[date_column].isna().sum()

          # Parse dates and identify invalid ones
          df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

          # Handle invalid dates based on strategy
          df, invalid_dates_report = handle_invalid_dates(df, date_column, imputation_strategy)

          # Apply timezone standardization
          if df[date_column].dt.tz is None:
              df[date_column] = df[date_column].dt.tz_localize('UTC')
          else:
              df[date_column] = df[date_column].dt.tz_convert('UTC')

          # Handle duplicate timestamps
          df, duplicates_report = handle_duplicate_timestamps(df, date_column)

          new_null_count = df[date_column].isna().sum()
          parsing_failures = new_null_count - original_null_count

          # Update report
          report[function_name] = True
          report[f"{function_name}_parsing_failures"] = int(parsing_failures)
          report[f"{function_name}_timezone"] = "UTC"
          report[f"{function_name}_invalid_dates"] = invalid_dates_report
          report[f"{function_name}_duplicates"] = duplicates_report

      except Exception as e:
          report[function_name] = False
          report[f"{function_name}_error"] = str(e)

      return df, report

def standardize_date_retain_timezone(df: pd.DataFrame, report: dict, date_column: str, imputation_strategy: str = "mean_timedelta") -> tuple[pd.DataFrame, dict]:
      """
      Standardize the date column format while retaining original timezone information.

      Parameters:
      -----------
      df : pd.DataFrame
          DataFrame containing the time series data
      report : dict
          Report dictionary to track transformations
      date_column : str
          Name of the date column to standardize
      imputation_strategy : str
          Strategy for handling invalid dates ("mean_timedelta", "leave_as_is", "auto_detected")

      Returns:
      --------
      tuple[pd.DataFrame, dict]
          Modified DataFrame and updated report dictionary
      """
      function_name = "standardize_date_retain_timezone"

      try:
          # Check if column exists
          if date_column not in df.columns:
              report[function_name] = False
              report[f"{function_name}_error"] = f"Column '{date_column}' not found"
              return df, report

          # Store original column info for reporting
          original_null_count = df[date_column].isna().sum()

          # Get a sample of non-null values to detect original timezone
          sample_values = df[date_column].dropna().head(10).astype(str).tolist()

          # Parse dates while preserving timezone information
          df[date_column] = pd.to_datetime(df[date_column], errors='coerce', utc=False)

          # Handle invalid dates based on strategy
          df, invalid_dates_report = handle_invalid_dates(df, date_column, imputation_strategy)

          # Handle duplicate timestamps
          df, duplicates_report = handle_duplicate_timestamps(df, date_column)

          # Count parsing failures
          new_null_count = df[date_column].isna().sum()
          parsing_failures = new_null_count - original_null_count

          # Determine timezone status for reporting
          if df[date_column].dt.tz is None:
              timezone_status = "timezone_naive"
          else:
              timezone_status = str(df[date_column].dt.tz)

          # Update report
          report[function_name] = True
          report[f"{function_name}_parsing_failures"] = int(parsing_failures)
          report[f"{function_name}_timezone_status"] = timezone_status
          report[f"{function_name}_sample_original_values"] = sample_values[:3]  # First 3 for reference
          report[f"{function_name}_invalid_dates"] = invalid_dates_report
          report[f"{function_name}_duplicates"] = duplicates_report

      except Exception as e:
          report[function_name] = False
          report[f"{function_name}_error"] = str(e)

      return df, report

def preview_date_transformation(df: pd.DataFrame, date_column: str, standardization_type: str) -> dict:
    """
    Preview date transformation without modifying the original data.

    Returns format detection, sample transformations, and validation warnings.
    """

    if date_column not in df.columns:
        return {
            "format_detection": None,
            "preview_samples": [],
            "validation_warnings": [f"Columna '{date_column}' no encontrada"]
        }

    # Get sample data (first 5 non-null values)
    sample_data = df[date_column].dropna().head(5).tolist()

    # Detect format and timezone
    format_detection = detect_date_format(sample_data)

    # Generate preview transformations
    preview_samples = []
    validation_warnings = []

    for i, original_value in enumerate(sample_data):
        try:
            if standardization_type == "utc":
                parsed_date = pd.to_datetime(original_value, errors='coerce')
                if pd.isna(parsed_date):
                    transformed = None
                elif parsed_date.tz is None:
                    transformed = parsed_date.tz_localize('UTC').isoformat()
                else:
                    transformed = parsed_date.tz_convert('UTC').isoformat()
            else:  # retain_timezone
                parsed_date = pd.to_datetime(original_value, errors='coerce', utc=False)
                transformed = parsed_date.isoformat() if not pd.isna(parsed_date) else None

            preview_samples.append({
                "original": str(original_value),
                "transformed": transformed
            })

            if transformed is None:
                validation_warnings.append(f"Fecha inválida encontrada: {original_value}")

        except Exception as e:
            preview_samples.append({
                "original": str(original_value),
                "transformed": None
            })
            validation_warnings.append(f"Error procesando fecha: {original_value}")

    return {
        "format_detection": format_detection,
        "preview_samples": preview_samples,
        "validation_warnings": validation_warnings
    }

def detect_date_format(sample_dates: list) -> dict:
    """
    Detect the most likely date format from sample data.
    """

    format_patterns = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S%z",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%Y/%m/%d"
    ]

    successful_parses = {}
    timezone_detected = None

    for date_str in sample_dates[:3]:  # Test first 3 samples
        for pattern in format_patterns:
            try:
                parsed = datetime.strptime(str(date_str), pattern)
                successful_parses[pattern] = successful_parses.get(pattern, 0) + 1
                break
            except ValueError:
                continue

        # Check for timezone info
        if '+' in str(date_str) or 'Z' in str(date_str) or any(tz in str(date_str) for tz in ['UTC', 'GMT']):
            timezone_detected = "Timezone detected"

    # Find most successful pattern
    best_pattern = max(successful_parses.items(), key=lambda x: x[1])[0] if successful_parses else "Mixed formats"
    success_rate = max(successful_parses.values()) / len(sample_dates) if successful_parses else 0

    return {
        "detected_format": best_pattern,
        "parsing_success_rate": success_rate,
        "timezone_info": timezone_detected or "No timezone detected"
    }

def handle_invalid_dates(df: pd.DataFrame, date_column: str, imputation_strategy: str) -> tuple[pd.DataFrame, dict]:
    """
    Handle invalid dates based on the specified strategy.
    """

    invalid_mask = df[date_column].isna()
    invalid_count = invalid_mask.sum()

    if invalid_count == 0:
        return df, {"invalid_dates_found": 0, "imputation_applied": False}

    if imputation_strategy == "leave_as_is":
        return df, {
            "invalid_dates_found": int(invalid_count),
            "imputation_applied": False,
            "strategy_used": "leave_as_is"
        }

    # Calculate timedelta for imputation
    valid_dates = df[date_column].dropna().sort_values()

    if len(valid_dates) < 2:
        return df, {
            "invalid_dates_found": int(invalid_count),
            "imputation_applied": False,
            "error": "Insufficient valid dates for timedelta calculation"
        }

    if imputation_strategy == "mean_timedelta":
        timedeltas = valid_dates.diff().dropna()
        mean_timedelta = timedeltas.mean()
    elif imputation_strategy == "auto_detected":
        mean_timedelta = detect_auto_timedelta(valid_dates)
    else:
        mean_timedelta = valid_dates.diff().dropna().mean()  # fallback

    # Apply imputation
    imputed_count = 0
    for idx in df[invalid_mask].index:
        # Find nearest valid dates
        prev_valid = df[date_column].iloc[:idx].dropna()
        next_valid = df[date_column].iloc[idx+1:].dropna()

        if len(prev_valid) > 0 and len(next_valid) > 0:
            # Interpolate between previous and next
            prev_date = prev_valid.iloc[-1]
            next_date = next_valid.iloc[0]
            df.loc[idx, date_column] = prev_date + (next_date - prev_date) / 2
            imputed_count += 1
        elif len(prev_valid) > 0:
            # Extrapolate from previous
            df.loc[idx, date_column] = prev_valid.iloc[-1] + mean_timedelta
            imputed_count += 1
        elif len(next_valid) > 0:
            # Extrapolate from next
            df.loc[idx, date_column] = next_valid.iloc[0] - mean_timedelta
            imputed_count += 1

    return df, {
        "invalid_dates_found": int(invalid_count),
        "imputation_applied": True,
        "dates_imputed": int(imputed_count),
        "strategy_used": imputation_strategy,
        "mean_timedelta_seconds": mean_timedelta.total_seconds() if pd.notna(mean_timedelta) else None
    }

def detect_auto_timedelta(valid_dates: pd.Series) -> pd.Timedelta:
    """
    Auto-detect the most likely timedelta pattern.
    """

    timedeltas = valid_dates.diff().dropna()

    # Common time intervals in seconds
    common_intervals = {
        3600: "hourly",      # 1 hour
        86400: "daily",      # 1 day
        604800: "weekly",    # 1 week
        2628000: "monthly",  # ~1 month
        31536000: "yearly"   # 1 year
    }

    # Find the most common timedelta
    timedelta_seconds = timedeltas.dt.total_seconds()
    mode_seconds = timedelta_seconds.mode()

    if len(mode_seconds) > 0:
        # Fix: Deterministic mode selection for reproducibility
        # When multiple modes exist, pandas.mode() returns them in arbitrary order
        # Always select the minimum mode value to ensure consistent behavior across runs
        mode_value = mode_seconds.min()
        if len(mode_seconds) > 1:
            logger.warning(f"Multiple modes found in timedelta: {list(mode_seconds)}. Using minimum: {mode_value}")

        # Check if it matches a common interval
        for seconds, name in common_intervals.items():
            if abs(mode_value - seconds) / seconds < 0.1:  # 10% tolerance
                return pd.Timedelta(seconds=seconds)

    # Fallback to mean
    return timedeltas.mean()

def handle_duplicate_timestamps(df: pd.DataFrame, date_column: str) -> tuple[pd.DataFrame, dict]:
    """
    Handle duplicate timestamps by adding microseconds.
    """

    duplicates_before = df[date_column].duplicated().sum()

    if duplicates_before == 0:
        return df, {"duplicates_found": 0, "duplicates_resolved": 0}

    # Add microseconds to duplicates
    df_sorted = df.sort_values(date_column)

    duplicate_mask = df_sorted[date_column].duplicated(keep='first')
    duplicate_indices = df_sorted[duplicate_mask].index

    for i, idx in enumerate(duplicate_indices):
        microseconds_to_add = (i + 1) * 1000  # Add 1ms, 2ms, etc.
        df.loc[idx, date_column] += pd.Timedelta(microseconds=microseconds_to_add)

    duplicates_after = df[date_column].duplicated().sum()

    return df, {
        "duplicates_found": int(duplicates_before),
        "duplicates_resolved": int(duplicates_before - duplicates_after)
    }

# FUNCTIONS FOR DETERMINING DATE INTERVAL REGULARITY #


def is_time_series_regular(csv_path, date_column):
    """
    Check if a time series dataset has regular intervals.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    date_column : str
        Name of the column containing dates/timestamps
        
    Returns:
    --------
    bool or None
        True if time series is regular, False if irregular, None for edge cases
    """
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if column exists
        if date_column not in df.columns:
            print(f"Column '{date_column}' not found in the dataset")
            return None
            
        # Filter out missing values
        date_series = df[date_column].dropna()
        
        # Edge case: less than 2 rows
        if len(date_series) < 2:
            return None
            
        # Auto-detect and parse dates
        parsed_dates = []
        for date_str in date_series:
            try:
                # Handle case where date might already be datetime
                if isinstance(date_str, (pd.Timestamp, datetime)):
                    parsed_dates.append(pd.to_datetime(date_str))
                else:
                    # Try multiple parsing strategies
                    parsed_date = pd.to_datetime(date_str)
                    parsed_dates.append(parsed_date)
            except:
                try:
                    # Fallback to dateutil parser
                    parsed_date = parser.parse(str(date_str))
                    parsed_dates.append(pd.to_datetime(parsed_date))
                except:
                    print(f"Could not parse date: {date_str}")
                    return None
        
        if len(parsed_dates) < 2:
            return None
            
        # Sort dates to ensure chronological order
        parsed_dates = sorted(parsed_dates)
        
        # Calculate intervals between consecutive dates
        intervals = []
        for i in range(1, len(parsed_dates)):
            interval = parsed_dates[i] - parsed_dates[i-1]
            intervals.append(interval)
        
        # Convert to seconds for easier comparison
        interval_seconds = [interval.total_seconds() for interval in intervals]
        
        # Detect frequency based on median interval
        median_seconds = np.median(interval_seconds)
        
        # Define frequency thresholds (in seconds)
        HOUR = 3600
        DAY = 86400
        MONTH_APPROX = 30.5 * DAY  # Average month length
        YEAR_APPROX = 365.25 * DAY  # Average year length
        
        # Determine frequency and apply appropriate strictness rules
        if median_seconds <= 2 * HOUR:  # Hourly or sub-hourly
            return _check_hourly_regularity(intervals, parsed_dates)
            
        elif median_seconds <= 2 * DAY:  # Daily
            return _check_daily_regularity(intervals)
            
        elif median_seconds <= 2 * MONTH_APPROX:  # Monthly
            return _check_monthly_regularity(parsed_dates)
            
        elif median_seconds <= 2 * YEAR_APPROX:  # Yearly
            return _check_yearly_regularity(intervals)
            
        else:
            # Intervals too large or inconsistent
            return False
            
    except Exception as e:
        print(f"Error processing file: {e}")
        return None


def _check_hourly_regularity(intervals, parsed_dates):
    """Check if hourly data is regular with ±5 minutes tolerance."""
    HOUR = timedelta(hours=1)
    TOLERANCE = timedelta(minutes=5)
    
    for interval in intervals:
        if abs(interval - HOUR) > TOLERANCE:
            return False
    return True


def _check_daily_regularity(intervals):
    """Check if daily data has exactly 1-day intervals."""
    DAY = timedelta(days=1)
    
    for interval in intervals:
        if interval != DAY:
            return False
    return True


def _check_monthly_regularity(parsed_dates):
    """Check if monthly data maintains same day-of-month."""
    # For monthly data, check if day of month is consistent
    # and months increment by 1 (accounting for year transitions)
    
    days_of_month = [date.day for date in parsed_dates]
    
    # Check if all dates have the same day of month
    if len(set(days_of_month)) > 1:
        return False
    
    # Check if months increment properly
    for i in range(1, len(parsed_dates)):
        prev_date = parsed_dates[i-1]
        curr_date = parsed_dates[i]
        
        # Calculate expected next month
        if prev_date.month == 12:
            expected_year = prev_date.year + 1
            expected_month = 1
        else:
            expected_year = prev_date.year
            expected_month = prev_date.month + 1
        
        if curr_date.year != expected_year or curr_date.month != expected_month:
            return False
    
    return True


def _check_yearly_regularity(intervals):
    """Check if yearly data has 365-366 day variations."""
    MIN_YEAR = timedelta(days=365)
    MAX_YEAR = timedelta(days=366)
    
    for interval in intervals:
        if interval < MIN_YEAR or interval > MAX_YEAR:
            return False
    return True


def is_time_series_regular_first_column(csv_path):
    """
    Check if a time series dataset has regular intervals using the first column as date column.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
        
    Returns:
    --------
    bool or None
        True if time series is regular, False if irregular, None for edge cases
    """
    
    try:
        # Read just the header to get the first column name
        df_header = pd.read_csv(csv_path, nrows=0)
        first_column = df_header.columns[0]
        
        # Call the main function with the first column name
        return is_time_series_regular(csv_path, first_column)
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

    
    # You can test with actual files like:
    # result = is_time_series_regular("data.csv", "date")
    # print(f"Time series regular: {result}")
    
    # Or using first column:
    # result = is_time_series_regular_first_column("data.csv")
    # print(f"Time series regular (first column): {result}")


###################################################################################
################################## MAIN FUNCTION ##################################
###################################################################################

def limpiar_datos(
    csv_input: str, 
    csv_output_eda: str,
    optional_methods: list
)-> dict:
    """
    """
    # Copy the input file to the output path as-is
    shutil.copy2(csv_input, csv_output_eda)
    
    # Read the data to get basic info for the report
    df = pd.read_csv(csv_input)
    
    # Return a basic report indicating no optional processing was done
    report = {
        "initial_rows": int(df.shape[0]),
        "initial_columns": list(df.columns),
        "final_rows": int(df.shape[0]),
        "final_columns": list(df.columns),
        "processing_type": "pass_through",
        "duplicates_removed": 0,
        "numeric_missing_before": {},
        "numeric_imputations": {},
        "categorical_missing_filled": {},
        "outliers_removed": {},
        "columns_removed_all_na": [],
        "converted_to_numeric": []
    }

    # ─────────────────────────────────────────────────────────────────────────────
    # Procesamiento obligatorio
    # ─────────────────────────────────────────────────────────────────────────────

    mandatory_methods= [
        lambda df, report: clean_column_names(df,report),
        lambda df, report: remove_whitespace_from_df_data(df,report),
        lambda df, report: replace_empty_values_with_nan(df,report),
        lambda df, report: convert_to_numeric_columns(df,report),
        lambda df, report: fill_categorical_missing(df,report),
        lambda df, report: remove_empty_columns(df,report),
    ]
    
    for method in mandatory_methods:
        df, report = method(df, report)

    # ─────────────────────────────────────────────────────────────────────────────
    # Procesamiento opcional
    # ─────────────────────────────────────────────────────────────────────────────
    
    def parse_optional_methods(optional_methods_input):
        """
        Parse optional methods dictionary into list of lambda functions.
        
        Args:
            optional_methods_input: Dict with 'cleaning_methods' key or JSON string containing method configs
            
        Returns:
            List of lambda functions that can be applied to df and report
        """
        if not optional_methods_input:
            return []
            
        # Handle case where input is a JSON string
        if isinstance(optional_methods_input, str):
            try:
                optional_methods_dict = json.loads(optional_methods_input)
            except json.JSONDecodeError:
                print(f"Error parsing JSON string: {optional_methods_input}")
                return []
        else:
            optional_methods_dict = optional_methods_input
            
        if 'cleaning_methods' not in optional_methods_dict:
            return []
            
        method_functions = []
        cleaning_methods = optional_methods_dict['cleaning_methods']
        
        for method_config in cleaning_methods:
            method_name = method_config['method']
            params = method_config.get('params', {})
            
            if method_name == 'drop_duplicates':
                include = params.get('include', True)
                method_functions.append(
                    lambda df, report, inc=include: drop_duplicates(df, report, inc)
                )
                
            elif method_name == 'fill_missing_numeric_values':
                method = params.get('method', 'media')
                value = params.get('value', None)
                method_functions.append(
                    lambda df, report, meth=method, val=value: fill_missing_numeric_values(df, report, meth, val)
                )
                
            elif method_name == 'filter_outliers':
                method_functions.append(
                    lambda df, report: filter_outliers(df, report)
                )
                
            elif method_name == 'standardize_date_to_utc':
                date_column = params.get('date_column')
                imputation_strategy = params.get('imputation_strategy', 'mean_timedelta')
                if date_column:
                    method_functions.append(
                        lambda df, report, col=date_column, strategy=imputation_strategy:
                        standardize_date_to_utc(df, report, col, strategy)
                    )

            elif method_name == 'standardize_date_retain_timezone':
                date_column = params.get('date_column')
                imputation_strategy = params.get('imputation_strategy', 'mean_timedelta')
                if date_column:
                    method_functions.append(
                        lambda df, report, col=date_column, strategy=imputation_strategy:
                        standardize_date_retain_timezone(df, report, col, strategy)
                    )
                
        return method_functions
    
    # Parse and apply optional methods
    if optional_methods:
        print("optional methods were passed...")
        print(optional_methods)
        optional_method_functions = parse_optional_methods(optional_methods)
        
        for method in optional_method_functions:
            df, report = method(df, report)
    else: 
        print("NO optional methods were passed...")
    # ─────────────────────────────────────────────────────────────────────────────
    # Pasos finales
    # ─────────────────────────────────────────────────────────────────────────────
    # # Guardar dataset limpio
    df.to_csv(csv_output_eda, index=False)
    report["final_rows"] = int(df.shape[0])
    report["final_columns"] = list(df.columns)

    # # Registrar si dataset quedo con intervalos temporales irregulares o no
    report["is_time_series_regular"] = is_time_series_regular_first_column(csv_output_eda)

    print(f"El dataset limpio fue guardado en: {csv_output_eda}")
    return report