import pandas as pd
import os
import glob

def merge_strain_and_tofmeta(strain_ed2p_df, strain_p2p_df, metadata_df):
    """
    Merges strain df with the dataset metadata."""

    if not os.path.exists(os.path.join(os.getcwd(),"strain_predictions", "merged_dfs")):
        os.mkdir(os.path.join(os.getcwd(),"strain_predictions", "merged_dfs"))
    
    print("Please ensure that the strain dataframes are specified to their corresponding parameter."
          + "\nIf they are, please ignore this.")
    
    assert all(strain_ed2p_df.pat.unique() == strain_p2p_df.pat.unique()), "Strain data frames must be from the same dataset."
    
    for strain_df,filename in zip([strain_ed2p_df, strain_p2p_df],["df_DMD_time_ed2p.csv","df_DMD_time_p2p.csv"]):
        
        all_patients = strain_df.pat.unique()

        print("Example patient ID before strain df formatting: ", strain_df.pat.unique()[0])

        ###Make patient ID pattern compatible with deformable registration model

        if hasattr(metadata_df,"PID"):
            patient_coloumn= "PID"
            diretory_path = os.path.join(os.getcwd(),"strain_predictions","merged_dfs","tof_gcn")
            if not os.path.exists(diretory_path):
                os.mkdir(diretory_path)
        else:
            patient_coloumn= "filename"
            diretory_path = os.path.join(os.getcwd(),"strain_predictions","merged_dfs","tof_indicator")
            if not os.path.exists(diretory_path):
                os.mkdir(diretory_path)

        len_before_marge = len(strain_df.pat.unique())

        metadata_df.loc[:,"pat"] = metadata_df[patient_coloumn].apply(lambda x: [p for p in all_patients if p.upper() in x.upper() or x.upper() in p.upper()][0])

        print("Example patient ID after strain df formatting: ", metadata_df.pat.unique()[0])

        strain_df = strain_df.merge(metadata_df, how="inner", on="pat")

        len_after_merge = len(strain_df["pat"].unique())

        assert len_before_marge == len_after_merge, "Merge failed, patient count is not consistent: before " + str(len_before_marge) +" vs after "+ str(len_after_merge)
        
        strain_df_path = os.path.join(diretory_path,filename)
        
        strain_df.to_csv(strain_df_path)

        
        
def standardize_strain_df(drop_unamed_columns=False):
    
    assert os.path.exists(os.path.join(os.getcwd(),"strain_predictions","merged_dfs","tof_indicator")) and os.path.exists(os.path.join(os.getcwd(),"strain_predictions","merged_dfs","tof_gcn")), "Please merge dataframes first before standardizing them."

    tof_indicator = glob.glob(os.path.join(os.getcwd(),"strain_predictions","merged_dfs","tof_indicator", "*.csv"))
    
    tof_gcn = glob.glob(os.path.join(os.getcwd(),"strain_predictions","merged_dfs","tof_gcn", "*.csv"))
    
    
    for strain_df_1_path,strain_df_2_path in zip(tof_indicator,tof_gcn):
        
        strain_df_1 = pd.read_csv(strain_df_1_path)
        strain_df_2 = pd.read_csv(strain_df_2_path)

        for column,tof_indicator_substitute in zip(["subjid","gender","decease"],["PID","History_Gender_geschlecht","Outcome y/n"]):
            if hasattr(strain_df_1,tof_indicator_substitute):
                strain_df_2 = strain_df_2.rename(columns={column:tof_indicator_substitute})
            elif hasattr(strain_df_2,tof_indicator_substitute):
                strain_df_1 = strain_df_1.rename(columns={column:tof_indicator_substitute})
            else:
                print("Target column not found: " + tof_indicator_substitute)

            ###Binarize the standardized columns if they have ""
            if issubclass(type(strain_df_2[tof_indicator_substitute].unique()[0]), str):
                if "Y" and "N" in [x.upper() for x in strain_df_2[tof_indicator_substitute].unique()]:
                    strain_df_2[tof_indicator_substitute] = strain_df_2[tof_indicator_substitute].apply(lambda x: 1 if x== "Y" else 0)
            if issubclass(type(strain_df_1[tof_indicator_substitute].unique()[0]), str):
                if "Y" and "N" in [x.upper() for x in strain_df_1[tof_indicator_substitute].unique()]:
                    strain_df_1[tof_indicator_substitute] = strain_df_1[tof_indicator_substitute].apply(lambda x: 1 if x== "Y" else 0)

        if drop_unamed_columns:                                           
            strain_df_2 = strain_df_2.drop([x for x in strain_df_2.columns if "Unnamed" in x], axis=1)
            strain_df_1 = strain_df_1.drop([x for x in strain_df_1.columns if "Unnamed" in x], axis=1)

        if not hasattr(strain_df_1,"phase"):
            strain_df_1 = strain_df_1.rename(columns= {"phase_x":"phase"})
        elif not hasattr(strain_df_2,"phase"):
            strain_df_2 = strain_df_2.rename(columns={"phase_x":"phase"})

        strain_df_2.to_csv(strain_df_2_path)
        strain_df_1.to_csv(strain_df_1_path)
        