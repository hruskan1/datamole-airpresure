import pandas as pd
import numpy as np
import pyarrow as pa

def get_presure_serie_for_machine_and_measurement(df, machine_id, measurement_id):
    """
    Function to get values for rows containing specified MachineId and MeasurmentId.
    
    Parameters:
        df (DataFrame): Input DataFrame.
        lf (DataFrame): Labels DataFrame.
        machine_id (int): MachineId value to filter on.
        measurement_id (int): MeasurementId value to filter on.
        
    Returns:
        numpy.array: Numpy array of values for rows containing specified MachineId and MeasurmentId.
    """
    # Filter rows based on MachineId and MeasurmentId in both df and lf
    filtered_df = df[(df['MachineId'] == machine_id) & (df['MeasurementId'] == measurement_id)]

    # Extract values from the filtered DataFrame and convert them to a numpy array
    values_array = filtered_df['Pressure']
    
    return values_array

def exponential_weighted_average(data, alpha):
    """
    Computes the Exponential Weighted Average (EWA) of the given data.
    
    Parameters:
        data (array-like): The input data as a 1-D array-like structure.
        alpha (float): The smoothing factor, often denoted as alpha, should be between 0 and 1.
        
    Returns:
        numpy.ndarray: The Exponential Weighted Average (EWA) of the input data.
    """
    # Initialize the EWA with the first data point
    ewa = np.zeros_like(data, dtype=float)
    ewa[0] = data[0]

    # Compute the EWA for subsequent data points
    for i in range(1, len(data)):
        ewa[i] = alpha * ewa[i - 1] + (1 - alpha) * data[i]

    return ewa


def trim_series(series,eps=0.01):
    """
    Trim a pandas Series by removing leading and trailing rows with all zero values.

    Parameters:
    - series: pandas Series to be trimmed
    - eps: float
    
    Returns:
    - trimmed_series: Trimmed pandas Series
    """
    # Find the index of the first non-zero value
    series = series.reset_index(drop=True)
    start_index = (series > eps).idxmax() if (series > eps).any() else 0

    # Find the index of the last non-zero value
    end_index = (series > eps)[::-1].idxmax() if (series > eps).any() else len(series)

    # Slice the Series to remove leading and trailing rows with all zero values
    trimmed_series = series.iloc[start_index:end_index+1]

    return trimmed_series

def trim_array(arr):
    """
    Trim the given PyArrow array by removing leading and trailing zeros.

    Parameters:
        arr (pyarrow.Array): The input PyArrow array.

    Returns:
        pyarrow.Array: The trimmed PyArrow array.
    """
    # Convert PyArrow array to NumPy array
    np_arr = arr.to_numpy()
    
    # Trim the NumPy array
    trimmed_np_arr = np.trim_zeros(np_arr)
    if len(trimmed_np_arr) == 0:
        raise ValueError(f"Array contains all zeros or was already empty")
    # Convert back to PyArrow array
    trimmed_pa_arr = pa.array(trimmed_np_arr)
    
    return trimmed_pa_arr

def cut_array(arr):
    """
    Cut the given PyArrow array by retaining only the first quarter of the elements.

    Parameters:
        arr (pyarrow.Array): The input PyArrow array.

    Returns:
        pyarrow.Array: The cut PyArrow array.
    """
    # Get the length of the array
    length = len(arr)
    
    # Calculate the index to cut at
    cut_index = length // 4
    
    # Slice the array
    cut_arr = arr.slice(0, cut_index)
    
    return cut_arr

def scale_array(arr,kth=2):
    """
    Scale the given PyArrow array by dividing it by the kthlargest element.

    Parameters:
        arr (pyarrow.Array): The input PyArrow array.

    Returns:
        pyarrow.Array: The scaled PyArrow array.
    """
    if len(arr) == 0:
        return arr
    # Convert PyArrow array to NumPy array
    np_arr = arr.to_numpy()
    
    # Find the index of the second largest element
    kth = min(len(arr),kth)
    i = np.argsort(np_arr)[-kth] 
    
    # Get the scaling factor
    scaling_factor = np_arr[i] 
    
    # Scale the array
    scaled_np_arr = np_arr / scaling_factor if scaling_factor > 0 else np_arr
    
    # Convert back to PyArrow array
    scaled_pa_arr = pa.array(scaled_np_arr)
    
    return scaled_pa_arr






