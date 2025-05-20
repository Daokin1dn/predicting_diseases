from disease_prediction_API import DiseasePredictionAPI

Disease_preding = DiseasePredictionAPI("data/dataset.csv", {"MLP" , "XGBoost" })
Disease_preding.load_data()
Disease_preding.train()
Disease_preding.predict()
Disease_preding.evaluate()
