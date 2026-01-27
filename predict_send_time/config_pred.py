# Zeppelin server address
ZEPP_URL = "http://150.6.14.94:30132"
NOTEBOOK_ID = "2MC68ADVY"

# Paragraph IDs (Pre)
# Paragraph 21: Load transformed data (transformers and transformed train/test data)
PARAGRAPH_IDS_PRE = [
    "paragraph_1764658338256_686533166",
    "paragraph_1764742922351_426209997",
    "paragraph_1764833771372_1110341451",
    "paragraph_1765521446308_1651058139",  # Paragraph 21: Pipeline Transformers and Transformed Data Loading
]

# Paragraph IDs (Main)
# Paragraphs 22-28: Model definition, training, prediction, and evaluation
PARAGRAPH_IDS = [
    "paragraph_1764836200898_700489598",   # Paragraph 22: ML Model Definitions (GBT, FM, XGBoost, LightGBM)
    "paragraph_1765789893517_1550413688",  # Paragraph 24: Click Prediction Model Training (XGBoost Classifier)
    "paragraph_1767010803374_275395458",   # Paragraph 25: Click-to-Action Gap Model Training (XGBoost Classifier)
    "paragraph_1765764610094_1504595267",  # Paragraph 26: Response Utility Regression Model Training (XGBoost Regressor)
    "paragraph_1765345345715_612147457",   # Paragraph 27: Model Prediction on Test Dataset
    "paragraph_1764838154931_1623772564",  # Paragraph 28: Click Model Performance Evaluation (Precision@K per Hour & MAP)
]

# Parameters
# No parameters needed for this workflow (no suffix filtering required)
PARAMS = []

# Spark restart options
RESTART_SPARK_AT_START = True  # Restart Spark before starting PRE paragraphs
RESTART_SPARK_AT_END = False    # Restart Spark after all paragraphs complete