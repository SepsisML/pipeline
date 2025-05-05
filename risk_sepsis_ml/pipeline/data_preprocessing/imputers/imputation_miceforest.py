import miceforest as mf


class MiceForestImputationStrategy:
    def __init__(self, dataframe, lab_attributes, vital_attributes):
        self.df = dataframe
        self.lab_attributes = lab_attributes
        self.vital_attributes = vital_attributes

    def impute(self, dataframe, lab_attributes, vital_attributes):
        lab_cols = lab_attributes
        vital_cols = vital_attributes

        dataframe[lab_cols] = dataframe[lab_cols].replace(-9999, np.nan)
        dataframe[vital_cols] = dataframe[vital_cols].replace(-9999, np.nan)
        # Create kernel for lab vars
        lab_attributes_kernel = mf.ImputationKernel(
            dataframe[lab_cols],
            random_state=1991
        )
        # Create kernel for vital vars
        vital_attributes_kernel = mf.ImputationKernel(
            dataframe[vital_cols],
            random_state=1991
        )

        # Run the MICE algorithm for 2 iterations
        lab_attributes_kernel.mice(2)
        vital_attributes_kernel.mice(2)

        # Return the completed dataset.
        dataframe[lab_cols] = lab_attributes_kernel.complete_data()
        dataframe[vital_cols] = vital_attributes_kernel.complete_data()
        self.write_collection(dataframe, "miceforest")
