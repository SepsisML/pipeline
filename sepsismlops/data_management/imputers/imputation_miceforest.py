import miceforest as mf
from mongo_utils import write_collection, load_collection

class MiceForestImputationStrategy:
    def __init__(
        self, 
        dataframe, 
        lab_attributes, 
        vital_attributes,
        load_from_db: bool = False,
        write_in_db: bool = False, 
        collection_name="imputation-miceforest",
        mongo_uri="mongodb://localhost:27017", 
        db_name="imputation"
    ):
        self.df = dataframe
        self.lab_attributes = lab_attributes
        self.vital_attributes = vital_attributes
        self.load_from_db = load_from_db
        self.write_in_db = write_in_db
        self.collection_name = collection_name
        self.mongo_uri = mongo_uri
        self.db_name = db_name

    def impute(self):
        if self.load_from_db:
            if not self.collection_name:
                raise ValueError("collection_name es requerido si load_from_db=True")
            return load_collection(self.mongo_uri, self.db_name, self.collection_name)

        self.miceforest_impute(self.df, self.lab_attributes, self.vital_attributes)
        return self.df
    
    def miceforest_impute(self, df, lab_attributes, vital_attributes):
        lab_cols = lab_attributes
        vital_cols = vital_attributes

        df[lab_cols] = df[lab_cols].replace(-9999, np.nan)
        df[vital_cols] = df[vital_cols].replace(-9999, np.nan)
        # Create kernel for lab vars
        lab_attributes_kernel = mf.ImputationKernel(
            df[lab_cols],
            random_state=1991
        )
        # Create kernel for vital vars
        vital_attributes_kernel = mf.ImputationKernel(
            df[vital_cols],
            random_state=1991
        )

        # Run the MICE algorithm for 2 iterations
        lab_attributes_kernel.mice(2)
        vital_attributes_kernel.mice(2)

        # Return the completed dataset.
        df[lab_cols] = lab_attributes_kernel.complete_data()
        df[vital_cols] = vital_attributes_kernel.complete_data()

        if self.write_in_db:
            write_collection(self.df, self.mongo_uri, self.db_name, self.collection_name)
