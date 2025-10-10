import numpy as np
import pandas as pd
from tifffile import imread
from skimage.measure import regionprops_table
import logging
import argparse
import sys
from pathlib import Path

# --- Logger Configuration --- #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


##################################################################################
# cell cycle label assignment (FUCCI-Red, FUCCI-Green, FUCCI-Negative, FUCCI-DP) #
##################################################################################

def _label_row(row):
    """
    Helper function to assign FUCCI class labels to a single row based on normalized intensities.
    
    Args:
        row: pandas Series
            A row from the dataframe containing red_int_norm, green_int_norm, and fucci_ratio
    
    Returns:
        str: FUCCI class label
    """
    # handle non-expressors first #
    if row['red_int_norm'] <= 1 and row['green_int_norm'] <= 1:
        return 'FUCCI-Negative'
    # then apply ratio thresholds #
    elif row['fucci_ratio'] > 1.1:         
        return 'FUCCI-Green'
    elif 0.9 <= row['fucci_ratio'] <= 1.1: 
        return 'FUCCI-Double'   
    elif row['fucci_ratio'] < 0.9:         
        return 'FUCCI-Red'
    else:                                  
        return np.nan


def assign_class_labels(df):
    """
    Assign FUCCI class labels to nuclei based on FUCCI-Red and FUCCI-Green intensities.
    
    Args:
        df: pandas DataFrame
            Nuclear features dataframe with Mean_FUCCI_Red_Intensity and Mean_FUCCI_Green_Intensity columns
    
    Returns:
        pandas DataFrame: Input dataframe with added FUCCI_Class column
    """
    logger.info("Assigning FUCCI class labels...")
    
    # generate columns for median-normalized red and green intensities and their ratio #
    df['red_int_norm'] = df['Mean_FUCCI_Red_Intensity'] / df["Mean_FUCCI_Red_Intensity"].median()
    df['green_int_norm'] = df['Mean_FUCCI_Green_Intensity'] / df["Mean_FUCCI_Green_Intensity"].median()
    df['fucci_ratio'] = df['green_int_norm'] / df['red_int_norm']

    # assign class labels #
    df['FUCCI_Class'] = df.apply(_label_row, axis=1)

    # drop intermediate columns #
    df = df.drop(columns=['red_int_norm', 'green_int_norm', 'fucci_ratio'])

    # move "FUCCI_Class" column before shcoeffs columns #
    shcoeff_cols = [col for col in df.columns if col.startswith("shcoeffs")]
    if shcoeff_cols:
        first_coeff_col_idx = df.columns.get_loc(shcoeff_cols[0])
        cols = list(df.columns)
        cols.remove("FUCCI_Class")
        cols.insert(first_coeff_col_idx, "FUCCI_Class")
        df = df[cols]
    
    logger.info(f"FUCCI class distribution:\n{df['FUCCI_Class'].value_counts()}")
    return df


###########################
# somite label assignment #
###########################

def assign_somite_labels(nuclear_df, somite_mask, num_somites_left, num_somites_right):
    """
    Assign somite labels to nuclei based on their spatial location within somite mask.
    
    Args:
        nuclear_df: pandas DataFrame
            Nuclear features dataframe with Centroid_Z, Centroid_Y, Centroid_X columns
        somite_mask: numpy array
            3D mask array where each somite is labeled with a unique integer ID
        num_somites_left: int
            Number of somites on the left side
        num_somites_right: int
            Number of somites on the right side
    
    Returns:
        pandas DataFrame: Input dataframe with added somite-related columns
    """
    logger.info("Assigning somite labels...")
    
    # create dataframe of somite metadata #
    somite_ids =             list(range(1, num_somites_left + num_somites_right + 1))
    somite_stages =          list(range(1, num_somites_left + 1)) + list(range(1, num_somites_right + 1))
    somite_axial_positions = list(range(num_somites_left, 0, -1)) + list(range(num_somites_right, 0, -1))
    somite_sides =           ["Left"] * num_somites_left + ["Right"] * num_somites_right

    somite_df = pd.DataFrame({
        "Somite_ID": somite_ids,
        "Somite_Stage": somite_stages,
        "Somite_Axial_Position": somite_axial_positions,
        "Somite_Side": somite_sides
    })

    # calculate somite centroids / volumes from mask --> merge with metadata df #
    logger.info("Calculating somite properties from mask...")
    somite_rp = pd.DataFrame(regionprops_table(somite_mask, properties=("label", "centroid", "area")))
    somite_rp.rename(columns={
        "label": "Somite_ID",
        "centroid-0": "Somite_Centroid_Z",
        "centroid-1": "Somite_Centroid_Y",
        "centroid-2": "Somite_Centroid_X",
        "area": "Somite_Volume"
    }, inplace=True)
    somite_df = somite_df.merge(somite_rp, on="Somite_ID")

    # calculate nuclear centroids at s3 resolution (downsampled by 8x) #
    logger.info("Mapping nuclei to somites...")
    centroids_s3 = pd.DataFrame({
        "z": (nuclear_df["Centroid_Z"] / 8).round().astype(int),
        "y": (nuclear_df["Centroid_Y"] / 8).round().astype(int),
        "x": (nuclear_df["Centroid_X"] / 8).round().astype(int)
    })
    centroids_s3["z"] = centroids_s3["z"].clip(0, somite_mask.shape[0] - 1)
    centroids_s3["y"] = centroids_s3["y"].clip(0, somite_mask.shape[1] - 1)
    centroids_s3["x"] = centroids_s3["x"].clip(0, somite_mask.shape[2] - 1)

    # generate somite IDs for each nucleus --> merge somite + nuclear dfs #
    somite_ids_per_nucleus = somite_mask[centroids_s3["z"], centroids_s3["y"], centroids_s3["x"]]
    nuclear_df["Somite_ID"] = somite_ids_per_nucleus
    nuclear_df["Somite_ID"] = nuclear_df["Somite_ID"].replace(0, np.nan)  # replace 0 with NaN to handle non-somite nuclei

    nuclear_df = nuclear_df.merge(
        somite_df, 
        on="Somite_ID",
        how="left"
    )
    
    # count nuclei assigned to somites #
    num_with_somites = nuclear_df["Somite_ID"].notna().sum()
    logger.info(f"Assigned {num_with_somites}/{len(nuclear_df)} nuclei to somites")
    
    return nuclear_df


#########################
# snakemake integration #
#########################

def main():
    """
    Main function for postprocessing nuclear features with FUCCI and optionally somite labels.
    """
    # --- Argument Parsing --- #

    parser = argparse.ArgumentParser( description="Postprocess nuclear features, assign FUCCI class and optionally somite labels.")
    # Input/Output Arguments
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file with nuclear features")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file with added labels")
    # Optional Arguments
    parser.add_argument("--enable_somite_analysis", action="store_true", help="Enable somite analysis (requires somite mask)")
    parser.add_argument("--somite_mask_path", type=str, default=None, help="Path to somite mask TIFF file (required if enable_somite_analysis is True)")
    parser.add_argument("--num_somites_left", type=int, default=None, help="Number of somites on the left side (required if enable_somite_analysis is True)")
    parser.add_argument("--num_somites_right", type=int, default=None, help="Number of somites on the right side (required if enable_somite_analysis is True)")

    # Check if running under Snakemake
    if 'snakemake' in globals():
        logger.info("Running under Snakemake, using snakemake object for parameters.")
        # Create a namespace object to mimic argparse result
        args = argparse.Namespace(
            input_csv=snakemake.input.csv,
            output_csv=snakemake.output.csv,
            enable_somite_analysis=snakemake.params.enable_somite_analysis,
            somite_mask_path=snakemake.params.get("somite_mask_path", None),
            num_somites_left=snakemake.params.get("num_somites_left", None),
            num_somites_right=snakemake.params.get("num_somites_right", None),
            log_dir=snakemake.params.log_dir
        )
    else:
        logger.info("Not running under Snakemake, parsing command-line arguments.")
        args = parser.parse_args()

    # --- Validation --- #
    logger.info(f"Loading nuclear features from: {args.input_csv}")
    if not Path(args.input_csv).exists():
        logger.error(f"Input CSV file not found: {args.input_csv}")
        sys.exit(1)
    
    if args.enable_somite_analysis:
        if not args.somite_mask_path:
            logger.error("Somite analysis enabled but no somite_mask_path provided")
            sys.exit(1)
        if not Path(args.somite_mask_path).exists():
            logger.error(f"Somite mask file not found: {args.somite_mask_path}")
            sys.exit(1)
        if args.num_somites_left in [None, 0] or args.num_somites_right in [None, 0]:
            logger.error("Somite analysis enabled but num_somites_left or num_somites_right not provided (must be > 0)")
            sys.exit(1)

    # --- Load data --- #
    nuclear_df = pd.read_csv(args.input_csv)
    logger.info(f"Loaded {len(nuclear_df)} nuclei from CSV")

    # --- Assign FUCCI class labels (always run) --- #
    nuclear_df = assign_class_labels(nuclear_df)

    # --- Assign somite labels (optional) --- #
    if args.enable_somite_analysis:
        logger.info(f"Loading somite mask from: {args.somite_mask_path}")
        somite_mask = imread(args.somite_mask_path)
        logger.info(f"Somite mask shape: {somite_mask.shape}, dtype: {somite_mask.dtype}")
        
        nuclear_df = assign_somite_labels(
            nuclear_df, 
            somite_mask, 
            args.num_somites_left, 
            args.num_somites_right
        )
        
        # drop unnecessary columns if they exist #
        cols_to_drop = ["Somite_Centroid_Z", "Somite_Centroid_Y", "Somite_Centroid_X",
                        "Chunk_Z", "Chunk_Y", "Chunk_X"]
        existing_cols_to_drop = [col for col in cols_to_drop if col in nuclear_df.columns]
        if existing_cols_to_drop:
            nuclear_df = nuclear_df.drop(columns=existing_cols_to_drop)
            logger.info(f"Dropped columns: {existing_cols_to_drop}")
    
    # reorder columns to put shcoeffs last #
    shcoeff_cols = [col for col in nuclear_df.columns if col.startswith("shcoeffs")]
    other_cols = [col for col in nuclear_df.columns if not col.startswith("shcoeffs")]
    nuclear_df = nuclear_df[other_cols + shcoeff_cols]

    # --- Save output --- #
    logger.info(f"Saving postprocessed features to: {args.output_csv}")
    nuclear_df.to_csv(args.output_csv, index=False)
    logger.info("Postprocessing complete!")


if __name__ == "__main__":
    main()